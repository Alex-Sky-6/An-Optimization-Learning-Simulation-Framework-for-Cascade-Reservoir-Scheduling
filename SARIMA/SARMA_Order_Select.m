function [AR_Order,MA_Order,SAR_Order,SMA_Order] = SARMA_Order_Select(data,max_ar,max_ma,max_sar,max_sma,season,di)

T = length(data);

for ar = 0:max_ar
    for ma = 0:max_ma
        for sar = 0:max_sar
            for sma = 0:max_sma
                if ar==0&&ma==0
                    infoC_Sum(ar+1,ma+1,sar+1,sma+1) = NaN;
                    continue
                end
                if sar==0&&sma == 0
                    infoC_Sum(ar+1,ma+1,sar+1,sma+1) = NaN;
                    continue
                end
                try
                    Mdl = creatSARIMA(ar,ma,sar,sma,season,di);
                    [~, ~, LogL] = estimate(Mdl, data, 'Display', 'off');
                    [aic,bic] = aicbic(LogL,(ar+ma+sar+sma+2),T); % In addition to ar and ma, there are constant term and variance +2
                    infoC_Sum(ar+1,ma+1,sar+1,sma+1) = bic+aic;   % Select the sum of BIC and AIC as the standard
                catch ME % Catch error information
                    msgtext = ME.message;
                    if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel'))
                         infoC_Sum(ar+1,ma+1,sar+1,sma+1) = NaN; % Cannot estimate parameters, directly assign nan
                        % msgtext = [msgtext,'  ','Unable to estimate ARIMA model parameters. This may be because the training data length is relatively small and requires a higher degree of parameter fitting. Please try reducing the values of max_ar and max_ma']
                    else
                        infoC_Sum(ar+1,ma+1,sar+1,sma+1) = NaN; % Cannot estimate parameters, directly assign nan
                        % msgbox(msgtext, 'Error')
                    end
                end
            end
        end
    end
end
ind = find(infoC_Sum==min(min(min(min(infoC_Sum)))));  % Find the minimum value index
[I1,I2,I3,I4] = ind2sub([max_ar+1,max_ma+1,max_sar+1,max_sma+1],ind);                 % Convert index to subscript
AR_Order  = I1 - 1;
MA_Order  = I2 - 1;
SAR_Order = I3 - 1;
SMA_Order = I4 - 1;
fprintf('\n=== SARIMA Parameter Selection Results ===\n');
for ar = 0:max_ar
    for ma = 0:max_ma
        for sar = 0:max_sar
            for sma = 0:max_sma
                score = infoC_Sum(ar+1, ma+1, sar+1, sma+1);
                if ~isnan(score)
                    fprintf('AR=%d, MA=%d, SAR=%d, SMA=%d, Score(AIC+BIC)=%.2f\n', ...
                        ar, ma, sar, sma, score);
                end
            end
        end
    end
end
end

