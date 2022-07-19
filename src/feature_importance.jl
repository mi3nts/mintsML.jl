using ShapML
using MLJ

"""
    predict_function(model, data)

A wrapper function that takes a trained MLJ model `model` and a DataFrame `data` of model features. The function returns a single column DataFrame with prediction values.
"""
function predict_function(model, data)
    return DataFrame(y_pred = predict(model, data))
end




"""
    getFeatureImportances(Xtest, mach, sample_size, min_allowed)

Get ranked feature importances using the mean absolute Shap value. Returns a sorted DataFrame with all features with relative importance above `min_allowed`
"""
function getFeatureImportances(Xtest, mach, sample_size, min_allowed)
    data_shap = ShapML.shap(
        explain = Xtest,
        model = mach,
        predict_function = predict_function,
        sample_size = sample_size,
        parallel = :features,
        seed=42,
    )

    # group by feature name
    g_datashap = groupby(data_shap, :feature_name)

    # aggregate by mean abs shap value
    data_plot = combine(g_datashap, :shap_effect => (x->mean(abs.(x))) => :mean_effect)

    # create relative importance
    data_plot.rel_importance = (data_plot.mean_effect)./maximum(data_plot.mean_effect)

    # sort by rel_importance
    sort!(data_plot, :rel_importance; rev=true)

    res_df = data_plot[data_plot.rel_importance .>= min_allowed, :]

    return res_df
end
