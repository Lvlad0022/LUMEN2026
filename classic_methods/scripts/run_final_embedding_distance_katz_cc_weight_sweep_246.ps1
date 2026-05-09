$ErrorActionPreference = "Stop"

$python = ".\classic_methods\.venv\Scripts\python.exe"
$runner = ".\classic_methods\scripts\run_final_embedding_distance_katz_datasets.py"
$datasets = @(
    "C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_2.csv",
    "C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_4.csv",
    "C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_6.csv"
)
$weights = @(0.1, 0.3, 0.6, 1.0)

foreach ($weight in $weights) {
    $weightLabel = ($weight.ToString("0.0", [System.Globalization.CultureInfo]::InvariantCulture)).Replace(".", "p")
    Write-Host ""
    Write-Host "=== Running processed_2/4/6 with customer_customer_weight=$weight ==="
    & $python $runner `
        --search-name "final_embedding_distance_katz_ccw_${weightLabel}_datasets_246" `
        --datasets $datasets `
        --customer-customer-weight $weight

    if ($LASTEXITCODE -ne 0) {
        throw "Customer-customer weight sweep failed for weight=$weight"
    }
}

Write-Host ""
Write-Host "All customer-customer weight sweep runs completed successfully."
