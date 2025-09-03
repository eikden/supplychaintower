# Supply Chain Control Tower Simulation

This project simulates a multi-agent supply chain control tower for a beverage dataset (47 SKUs) with:

- ForecastingAgent: per-SKU gradient boosting (weekly aggregation, seasonality, trend, lags)
- InventoryAgent: ROP + EOQ policies with ABC–XYZ driven service levels
- PMAgentLLM: (optional) policy adjustment via OpenAI (falls back to rules if key/model unavailable)

## Files
- `supplychaintower.py` – main simulation script
- `beverage_sales.csv` – sample dataset (Order line items)
- `requirements.txt` – Python dependencies (baseline)
containing your OpenAI API key

## Recent Fixes
- Repaired corrupted duplicate `detect_columns` / `load_dataset` definitions
- Added robust column detection with explicit overrides (`--date-col`, `--sku-col`, `--qty-col`)
- Relaxed date parsing: drops a small number of bad rows instead of hard failing
- Corrected malformed import (`matplotlib.pyplot as plt os`)
- Implemented proper `if __name__ == "__main__"` block and argument forwarding
- Ensured Baseline scenario no longer applies dynamic PM adjustments


## Install
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
(If `python-docx`, `langchain`, or `langchain-openai` fail, upgrade pip: `python -m pip install --upgrade pip`.)

## Run
Explicitly pass the correct columns (recommended):
```powershell
python supplychaintower.py --data beverage_sales.csv --outdir out --date-col Order_Date --sku-col Product --qty-col Quantity
```

Progress bars (tqdm) show per-scenario SKU-week processing automatically if `tqdm` is installed (it is listed in requirements). Disable with `--no-progress`.

Outputs written to `out/`:
- `kpis_summary.csv`
- `segment_performance.csv`
- `sim_details_<Scenario>.csv`
- `plot_topA_forecast.png`
- (Optional) `Results.docx` if `python-docx` installed

## OpenAI Key
```powershell
$env:OPENAI_API_KEY = "sk-..."
```
The script will use rule-based PM decisions if no key or if LangChain cannot initialize the model.

## How to run locally

Save the file below as supplychaintower.py

```
pip install -U pandas numpy scikit-learn scipy matplotlib python-docx langchain langchain-openai openai
```

Put your data (Excel/CSV) in the same folder (e.g., beverage_sales.xlsx)

Set key & (optionally) model:
```
mac/linux: export OPENAI_API_KEY="sk-..."; export OPENAI_MODEL="gpt-5"

windows (powershell): [Environment]::SetEnvironmentVariable("OPENAI_API_KEY","sk-...","User")
```

```
Run: python supplychaintower.py --data beverage_sales.xlsx --outdir ./out
```

## Notes / Next Steps
- Consider adding unit tests for column detection and inventory policy
- Pin dependency versions for reproducibility
- Add CLI option for lead time and cost parameters
- Add seed control for price generation in segmentation

## Disclaimer
Do **not** commit real API keys. Rotate any exposed key immediately.
