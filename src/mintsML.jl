module mintsML

using Plots, StatsPlots

# Write your package code here.
include("plot_defaults.jl")
include("mints_recipes.jl")
include("utils.jl")


export add_mints_theme
export makeDatasets
export r²
export quantilequantile


end
