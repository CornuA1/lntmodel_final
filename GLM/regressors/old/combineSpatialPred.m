function combineSpatialPred()
% combine spatial predictors
dual_RF_predictors = zeros(size(predictor_short_x,1),size(predictor_short_x,2) * size(predictor_short_x,2));
predictor_short_x_T = transpose(predictor_short_x);
i = 1;
for x1 = 1: size(predictor_short_x_T,1)
    for y1 = 1: size(predictor_short_x_T,2)
        npx1 = predictor_short_x_T(x1,y1);
        for x2 = 1: size(predictor_short_x_T,1)
            for y2 = 1:size(predictor_short_x_T,2)
                npx2 = predictor_short_x_T(x2,y2);
                dual_RF_predictors(:,i) = npx1 + npx2;
                i = i + 1;
            end
        end
    end
end
end