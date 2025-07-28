# Data Generation Instructions

1. **Download the Dataset:**  
    Access the data via [Google Drive - Network Analysis Lattes Data](https://drive.google.com/drive/folders/15l9MfqX2i0WNELkWVCTkmUDjSzfsSEt8?usp=sharing).

    > **Note:** Skip downloading files from the `cabecalhos` (headers) folder.

2. **Extract Files:**  
    Unzip all downloaded directories into `data/raw/`.  
    Ensure the extracted folders are at the same level as the `headers` directory.

3. **Prepare Data:**  
    Execute all cells in `src/data/add_header.ipynb`.  
    *(This notebook will be converted to a Python script in future updates.)*

    - **Python version required:** 3.12.7

---

## Chart Generation Instructions

1. Open and run `notebooks/generic_dataset_analysis.ipynb`.

2. Generated charts will be saved automatically in `reports/analysis/`.

---

## Dependency Installation

- All required Python packages are installed using [uv](https://github.com/astral-sh/uv).
- To install dependencies, run:
  ```bash
  uv pip install -r requirements.txt
  ```

---

## Additional Notes

- Ensure all dependencies are installed before running the notebooks.
- For troubleshooting or questions, please refer to the repository's issues section.