using mintsML
using Documenter

DocMeta.setdocmeta!(mintsML, :DocTestSetup, :(using mintsML); recursive=true)

makedocs(;
    modules=[mintsML],
    authors="John Waczak",
    repo="https://github.com/mi3nts/mintsML.jl/blob/{commit}{path}#{line}",
    sitename="mintsML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mi3nts.github.io/mintsML.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mi3nts/mintsML.jl",
    devbranch="main",
)
