using PlotThemes
using PlotUtils

mints_palette = [
    colorant"#3cd184", # mint green
    colorant"#f97171", # dark coral
    colorant"#1e81b0", # dark blue
    colorant"#66beb2", # dark blue-green
    colorant"#f99192", # light coral
    colorant"#8ad6cc", # middle blue-green
    colorant"#3d6647", # dark green
    #        colorant"#8FDDDF", # middle blue
]





function add_mints_theme()
    _mints = PlotThemes.PlotTheme(Dict([
        # :background => :white,
        :framestyle => :box,
        :grid => true,
        :gridalpha => 0.4,
        :linewidth => 1.5,
        :markerstrokewidth => 0,
        :fontfamily => "Computer Modern",
        :colorgradient => :vik,  # or :turbo for Dr. Lary's preference
        :guidefontsize => 12,
        :titlefontsize => 12,
        :tickfontsize => 8,
        :palette => mints_palette,
        :minorgrid => true,
        :minorticks => 5,
        :gridlinewidth => 0.7,
        :minorgridalpha => 0.15,
        :legend => :outertopright
    ]))

    PlotThemes.add_theme(:mints, _mints)
end

