import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import time
import functools

# This experiment integrates a benchmark from the Assistax library.
# It assumes `assistax` is installed as a dependency.
from assistax.envs.registration import make_env
from assistax.wrappers.aht import AdHocTeamwork


class ActorCritic(nn.Module):
    """A simple Actor-Critic network for the PPO agent."""

    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation_fn = nn.relu
        else:
            activation_fn = nn.tanh

        actor_mean = nn.Dense(
            256,
            kernel_init=nn.initializers.orthogonal(2),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        actor_mean = activation_fn(actor_mean)
        actor_mean = nn.Dense(
            256,
            kernel_init=nn.initializers.orthogonal(2),
            bias_init=nn.initializers.constant(0.0),
        )(actor_mean)
        actor_mean = activation_fn(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256,
            kernel_init=nn.initializers.orthogonal(2),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        critic = activation_fn(critic)
        critic = nn.Dense(
            256,
            kernel_init=nn.initializers.orthogonal(2),
            bias_init=nn.initializers.constant(0.0),
        )(critic)
        critic = activation_fn(critic)
        critic = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


def make_train(config: DictConfig):
    """Factory for the PPO train function."""
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env, env_params = make_env(config["ENV_NAME"], num_envs=config["NUM_ENVS"])
    env = AdHocTeamwork(
        env=env,
        zoo_path=config["ZOO_PATH"],
        num_train_partners=config["NUM_TRAIN_PARTNERS"],
        num_test_partners=config["NUM_TEST_PARTNERS"],
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LEARNING_RATE"] * frac

    def train(rng):
        # INIT NETWORK
        # NOTE: We are training 'agent_0', the robot.
        agent_action_space = env.action_space(env_params, "agent_0")
        agent_observation_space = env.observation_space(env_params, "agent_0")
        network = ActorCritic(
            agent_action_space.shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *agent_observation_space.shape))
        network_params = network.init(_rng, init_x)["params"]
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(
                learning_rate=(
                    linear_schedule if config["ANNEAL_LR"] else config["LEARNING_RATE"]
                ),
                eps=1e-5,
            ),
        )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, _):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs["agent_0"])
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # The AHT wrapper requires a dict of actions
                actions = {"agent_0": action}
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, actions, env_params
                )
                transition = (
                    last_obs["agent_0"],
                    action,
                    log_prob,
                    done["__all__"],
                    reward["agent_0"],
                    value,
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGES
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs["agent_0"])

            def _calculate_gae(traj_batch, last_val):
                # ... GAE logic implementation ...
                return advantages, targets

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                # ... PPO update logic ...
                return update_state, (
                    total_loss,
                    (loss_actor, loss_critic, loss_entropy),
                )

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            # METRIC LOGGING
            metrics = traj_batch[-1]  # info
            runner_state = (update_state[0], env_state, last_obs, rng)
            return runner_state, metrics

        runner_state = (train_state, env_state, obsv, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point to run the ZSC benchmark experiment."""
    config = OmegaConf.to_container(cfg, resolve=True)
    print("--- Starting Assistax ZSC Benchmark ---")
    print(OmegaConf.to_yaml(cfg))

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))

    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()

    print(f"\n--- Benchmark Complete ---")
    print(f"Total time: {end_time - start_time:.2f}s")

    # Save results
    save_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    # A real implementation would save model params and detailed metrics.
    # For this proposal, we just confirm completion.
    (save_path / "success.txt").write_text("ZSC benchmark completed successfully.")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
