# Sample Transform Cost Hallucination Detection

Core implementation of *Sample Transform Cost-Based Training-Free Hallucination Detector for Large Language Models* (NeurIPS 2026).

Python 3.10-3.12:

```bash
pip install -r requirements.txt
python main.py
```

```python
from detector import read_data, score

questions, embeddings = read_data("example_data.npz")
avgwd, eigenwd = score(embeddings[0])
print(f"AvgWD: {avgwd:.6f}, EigenWD: {eigenwd:.6g}")
```

`score` takes a list of response-token arrays `(tokens, dim)`, excluding prompt, EOS, and padding. `project_responses` provides shared Gaussian projection.

Example: five [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) questions with pre-extracted 128D features from [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). Runs on CPU without downloading a model. Provenance is stored in the NPZ `metadata` field.

Code: [MIT](LICENSE). Example data: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), adapted from SQuAD 2.0 (Rajpurkar, Jia, and Liang) with responses and features added for demonstration.
