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

deploydocs(
    repo = "github.com/exanauts/ExaTron.jl.git",
    devbranch = "sc2021",
)
