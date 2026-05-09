$ErrorActionPreference = "Stop"

$python = ".\classic_methods\.venv\Scripts\python.exe"
$runner = ".\classic_methods\scripts\run_final_embedding_distance_katz_datasets.py"
$neighbors = @(5, 10, 20)

foreach ($n in $neighbors) {
    Write-Host ""
    Write-Host "=== Running final embedding-distance Katz with n_neighbors=$n ==="
    & $python $runner `
        --search-name "final_embedding_distance_katz_neighbors_$n" `
        --n-neighbors $n

    if ($LASTEXITCODE -ne 0) {
        throw "Final Katz neighbor sweep failed for n_neighbors=$n"
    }
}

Write-Host ""
Write-Host "All final Katz neighbor sweep runs completed successfully."
