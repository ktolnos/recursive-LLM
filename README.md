# recursive-LLM

## Supported Datasets

This script supports multiple-choice question answering datasets. The data processing logic adapts based on the `dataset_name` argument.

### AI2 ARC

-   **Hugging Face ID**: `allenai/ai2_arc`
-   **Configuration**: `ARC-Challenge`
-   **Splits**: `train`, `test`, `validation`
-   **Example usage**:
    ```bash
    python main.py --dataset_name allenai/ai2_arc --dataset_config ARC-Challenge --split test
    ```

### MMLU

-   **Hugging Face ID**: `cais/mmlu`
-   **Configuration**: Can be a specific subject (e.g., `abstract_algebra`) or `all` to use all subjects.
-   **Splits**: `test`, `dev`, `validation`, `auxiliary_train`
-   **Example usage**:
    ```bash
    python main.py --dataset_name cais/mmlu --dataset_config all --split test
    ```

### MMLU-Pro

-   **Hugging Face ID**: `TIGER-Lab/MMLU-Pro`
-   **Configuration**: This dataset does not have configurations.
-   **Splits**: `test`, `validation`
-   **Example usage**:
    ```bash
    python main.py --dataset_name TIGER-Lab/MMLU-Pro --dataset_config default --split test
    ```
