# Customization Guide

This guide explains how to extend the repository by adding your own models and verification algorithms. 

## Adding Your Own Model

Models in this repository are dynamically loaded through the `registry.yaml` file. The inference runtime uses a `ModelBroker` to interact with your model implementation. This setup allows you to plug in any model without modifying the core runtime code.

### 1. Update `registry.yaml`

To add a new model, create an adapter script in the `adapters/` directory and register it under the `available_models` section in `registry.yaml`.

Example entry:
```yaml
available_models:
  - name: "my_custom_model"
    path: "./adapters/my_adapter.py"
    init_func: "load_my_model"
    tokenizer_func: "my_tokenize"
    gen_func: "my_generate"
    model_id: "Author/My-Custom-Model"
```

### 2. Implement the Adapter Functions

Your adapter file (e.g., `./adapters/my_adapter.py`) must implement the three functions specified in the configuration:

#### The Initialization Function
Responsible for loading the model and tokenizer into memory.
```python
def load_my_model(model_id, **kwargs):
    # Load your model and tokenizer based on 'model_id'
    model = ...
    tokenizer = ...
    return {"model": model, "tokenizer": tokenizer} # This becomes 'model_state'
```

#### The Tokenizer Function
Responsible for converting text prompts into token IDs.
```python
def my_tokenize(prompt, model_state, **kwargs):
    tokenizer = model_state["tokenizer"]
    return tokenizer(prompt, return_tensors="pt")
```

#### The Generate Function
Responsible for performing the forward pass.
```python
def my_generate(input_toks, model_state, **kwargs):
    model = model_state["model"]
    outputs = model(**input_toks)
    return outputs.logits
```

### 3. Adding Custom Kwargs

Any extra flags passed via the command line (e.g., `--temperature 0.8 --custom_flag True`) are automatically parsed into a dictionary of `custom_kwargs`. The `ModelBroker` acts as a smart filter: it uses Python's `inspect` module to check what keyword arguments your functions accept and dynamically passes only the relevant kwargs to each function (unless your function accepts `**kwargs` explicitly, in which case it receives everything).

## Adding Your Own Verification Algorithm

Verification algorithms control the iterative decoding process (such as handling masking/unmasking or speculative acceptance).

### 1. Update `registry.yaml`

To add a custom algorithm, add it under `available_algorithms` in `registry.yaml`:

```yaml
available_algorithms:
  - name: "my_custom_algorithm"
    path: "./algorithms/my_algorithm.py"
    step_func: "run_step"
```

### 2. Implement the Step Function

Your algorithm file must implement the registered step function. The runtime iterates over `total_steps`, calling your function for each step. 

#### Required Inputs

- `drafter`: The `ModelBroker` instance for the drafting model.
- `verifier`: The `ModelBroker` instance for the verification model.
- `prompts`: A list of input prompt strings.
- `step_idx`: The current iteration step index (from 0 to `total_steps - 1`).
- `total_steps`: The total number of steps configured for the run.
- `algo_state`: A generic state object (dictionary) that is passed from one step to the next to maintain arbitrary state. It is empty (`{}`) on `step_idx == 0`.
- `**kwargs`: Any additional parsed arguments (e.g., `--gen_length`, `--mask_id`, and other CLI flags).

#### Expected Output

Your function should return a dictionary containing:
- `algo_state`: The updated state dictionary to pass into the next iteration.
- `metrics`: A dictionary containing performance measurements for the step (e.g., `verifier_nfe`, `drafter_nfe`).

**On the last step** (`step_idx == total_steps - 1`), the returned dictionary must also include a `results` key, which contains the final decoded string outputs.

#### Example Signature

```python
def run_step(drafter, verifier, prompts, step_idx, total_steps, algo_state, **kwargs):
    # Step 1. First step initialization
    if step_idx == 0:
        # Initialize content, apply initial mask
        algo_state["content"] = ...

    # Step 2. Generate and Verify
    # Consult drafter and verifier models using drafter.generate() / verifier.generate()

    # Step 3. Compute Metrics
    metrics = {"verifier_nfe": 1, "drafter_nfe": 1}

    # Step 4. Construct Output
    result = {
        "algo_state": algo_state,
        "metrics": metrics
    }

    # Step 5. Final Step Decoding
    if step_idx == total_steps - 1:
        # result["results"] = ... list of final strings
        pass

    return result
```