using Documenter, ExaTron

makedocs(
    sitename="ExaTron.jl",
    pages = [
        "Home" => "index.md",
        "gettingstarted.md",
        "Use Cases" => [
            "admm.md"
        ],
        "api.md"
    ]
)
