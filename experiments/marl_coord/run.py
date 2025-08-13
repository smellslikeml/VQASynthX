import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    num_agents: int = 3
    total_timesteps: int = 50000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        pi = distrax.Categorical(logits=logits)
        return pi


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return value.squeeze(-1)


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"]

    env = jaxmarl.make("MPE_simple_spread_v3", num_agents=config["NUM_AGENTS"])
    env = LogWrapper(env)

    def train(rng):
        # INIT NETWORKS
        actor = Actor(action_dim=env.action_space(env.agents[0]).n)
        critic = Critic()

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, env.observation_space(env.agents[0]).shape[0]))
        actor_params = actor.init(_rng, init_x)["params"]

        rng, _rng = jax.random.split(rng)
        # The critic uses the global state (concatenated obs) as in CTDE
        global_state_shape = (
            1,
            env.observation_space(env.agents[0]).shape[0] * config["NUM_AGENTS"],
        )
        init_x_critic = jnp.zeros(global_state_shape)
        critic_params = critic.init(_rng, init_x_critic)["params"]

        # INIT OPTIMIZER
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(config["LEARNING_RATE"], eps=1e-5),
        )

        actor_train_state = TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=tx
        )
        critic_train_state = TrainState.create(
            apply_fn=critic.apply, params=critic_params, tx=tx
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, state = jax.vmap(env.reset)(jax.random.split(_rng, config["NUM_ENVS"]))

        # TRAINING LOOP
        def _update_step(runner_state, _):
            actor_ts, critic_ts, obsv, env_state, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(carry, _):
                actor_ts, obsv, env_state, rng = carry
                rng, _rng = jax.random.split(rng)

                # Decentralized action selection
                pi = jax.vmap(actor.apply, in_axes=(None, 0))(actor_ts.params, obsv)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Centralized value estimation
                global_state = obsv.reshape(config["NUM_ENVS"], -1)
                value = critic.apply(critic_ts.params, global_state)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv_prime, state_prime, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, action
                )

                # Store experience (obs, action, log_prob, value, reward, done)
                transition = (
                    obsv,
                    action,
                    log_prob,
                    value,
                    jnp.mean(reward, axis=1),
                    jnp.all(done, axis=1),
                )
                return (actor_ts, obsv_prime, state_prime, rng), transition

            (actor_ts, obsv, env_state, rng), traj_batch = jax.lax.scan(
                _env_step, (actor_ts, obsv, env_state, rng), None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE (GAE)
            global_state = obsv.reshape(config["NUM_ENVS"], -1)
            last_val = critic.apply(critic_ts.params, global_state)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_val, transition):
                    gae, next_val = gae_and_next_val
                    _, _, _, val, reward, done = transition
                    delta = reward + config["GAMMA"] * next_val * (1 - done) - val
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, val), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                )
                return (
                    advantages,
                    advantages + traj_batch[3],
                )  # advantages + values = returns

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minibatch(train_states, batch_info):
                    actor_ts, critic_ts = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(params, traj_batch, gae):
                        obsv, action, log_prob, _, _, _ = traj_batch
                        pi = actor.apply(params, obsv)
                        log_prob_new = pi.log_prob(action)

                        ratio = jnp.exp(log_prob_new - log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss1 = ratio * gae
                        loss2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        actor_loss = -jnp.minimum(loss1, loss2).mean()

                        entropy = pi.entropy().mean()
                        return actor_loss - config["ENT_coef"] * entropy

                    def _critic_loss_fn(params, traj_batch, targets):
                        obsv, _, _, _, _, _ = traj_batch
                        global_state = obsv.reshape(config["MINIBATCH_SIZE"], -1)
                        values_pred = critic.apply(params, global_state)
                        return jnp.square(values_pred - targets).mean()

                    # Actor update
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_ts.params, traj_batch, advantages
                    )
                    actor_ts = actor_ts.apply_gradients(grads=actor_grads)

                    # Critic update (on global state)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_ts.params, traj_batch, targets
                    )
                    critic_ts = critic_ts.apply_gradients(grads=critic_grads)

                    return (actor_ts, critic_ts), (actor_loss, critic_loss)

                (actor_ts, critic_ts), (actor_loss, critic_loss) = _update_minibatch(
                    (actor_ts, critic_ts), (traj_batch, advantages, targets)
                )
                return (actor_ts, critic_ts), (actor_loss, critic_loss)

            (actor_ts, critic_ts), (actor_loss, critic_loss) = _update_epoch(
                (actor_ts, critic_ts),
                None,
            )

            metric = info["returned_episode_returns"]
            runner_state = (actor_ts, critic_ts, obsv, env_state, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (actor_train_state, critic_train_state, obsv, state, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    args = tyro.cli(Args)
    config = {
        "LEARNING_RATE": args.learning_rate,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_AGENTS": args.num_agents,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "ENT_coef": args.ent_coef,
        "VF_coef": args.vf_coef,
    }

    rng = jax.random.PRNGKey(args.seed)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    output = train_jit(rng)

    print(f"Average return: {output['metrics'].mean()}")
