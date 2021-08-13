    function out = glmRun(reg2fit, reg2test, sig2fit, sig2test, fit_type, options)

    % apply glmnet
    if strcmp(fit_type,'gaussian')
        fit = glmnet(reg2fit,sig2fit, 'gaussian', options);
    elseif strcmp(fit_type,'poisson')
        fit = glmnet(reg2fit,sig2fit, 'poisson', options);
    end
     
    out.coef = coef;
    out.r2 = r2Val;
end