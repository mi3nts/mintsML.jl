using mintsML
using Documenter

DocMeta.setdocmeta!(mintsML, :DocTestSetup, :(using mintsML); recursive=true)

makedocs(;
    modules=[mintsML],
    authors="John Waczak",
    repo="https://github.com/john-waczak/mintsML.jl/blob/{commit}{path}#{line}",
    sitename="mintsML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://john-waczak.github.io/mintsML.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/john-waczak/mintsML.jl",
    devbranch="main",
)
