using Statistics


"""
Compute the r² value between two vectors.
"""
function r²(y_pred, y_true)
    ȳ = mean(y_true)
    SS_res = sum([(y_true[i]-y_pred[i])^2 for i ∈ 1:length(y_pred)])
    SS_tot = sum([(y_true[i]-ȳ)^2 for i ∈ 1:length(y_pred)])
    return 1 - SS_res/(SS_tot + eps(eltype(y_pred)))
end


