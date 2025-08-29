JUDGE_PROMPT = """
As an expert evaluator, your task is to assess a model's response to a spatial reasoning question about an image. You will be given the question, the model's generated answer, and a ground truth answer for reference.

Evaluate the model's answer on a scale of 0 to 10, where 0 is completely incorrect and 10 is perfect. Consider factors like correctness, relevance, and accuracy of spatial relationships and measurements.

Provide ONLY an integer score from 0 to 10 and nothing else.

---

Question: {question}
Ground Truth Answer: {ground_truth_answer}
Model's Answer: {model_answer}

---

Your Score (0-10):"""
