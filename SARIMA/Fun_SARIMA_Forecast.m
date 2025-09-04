function [forData,lower,upper] = Fun_SARIMA_Forecast(data,step,max_ar,max_ma,max_sar,max_sma,S,figflag)
%% 1. Data preprocessing
data = data(:);  % Unify as column vector
%% 2. Determine differencing order, non-seasonal differencing order D takes default value 1, d loops from 0 to 3 until stationarity is achieved
for d = 0:3
    D1 = LagOp({1 -1},'Lags',[0,d]);     % Non-seasonal differencing operator
    D12 = LagOp({1 -1},'Lags',[0,1*S]);  % Seasonal differencing operator
    D = D1*D12;          % Combination
    dY = filter(D,data); % Apply differencing to original data
    if(getStatAdfKpss(dY)) % Test for stationarity
        disp(['Non-seasonal differencing order is ',num2str(d),', seasonal differencing order is 1']);
        break;
    end
end
%% 3. Determine parameters ARlags, MALags, SARLags, SMALags
% Plot ACF and PACF
figure('Name','Stationary signal autocorrelation plot','Visible',figflag)
autocorr(dY)
figure('Name','Stationary signal partial autocorrelation plot','Visible',figflag)
parcorr(dY)
% Use AIC/BIC to determine parameters
try
    [AR_Order,MA_Order,SAR_Order,SMA_Order] = SARMA_Order_Select(dY,max_ar,max_ma,max_sar,max_sma,S,d); % Automatic parameter selection
catch ME % Catch error information
    msgtext = ME.message;
    if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel'))
         msgtext = [msgtext,'  ','Unable to estimate ARIMA model parameters. This may be because the training data length is relatively small and requires a higher degree of parameter fitting. Please try reducing the values of max_ar, max_ma, max_sar, max_sma'];
    end
    msgbox(msgtext, 'Error')
end
disp(['ARlags=',num2str(AR_Order),',MALags=',num2str(MA_Order),',SARLags=',num2str(SAR_Order),',SMALags=',num2str(SMA_Order)]);
%% 4. Parameter estimation
Mdl = creatSARIMA(AR_Order,MA_Order,SAR_Order,SMA_Order,S,d);  % Create SARIMA model
try
    EstMdl = estimate(Mdl,data);
catch ME % Catch error information
    msgtext = ME.message;
    if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel'))
         msgtext = [msgtext,'  ','Unable to estimate ARIMA model parameters. This may be because the training data length is relatively small and requires a higher degree of parameter fitting. Please try reducing the values of max_ar and max_ma']
    end
    msgbox(msgtext, 'Error')
    return
end
[res,~,logL] = infer(EstMdl,data);   % res is residual

stdr = res/sqrt(EstMdl.Variance);
figure('Name','Residual analysis','Visible',figflag)
subplot(2,3,1)
plot(stdr)
title('Standardized Residuals')
subplot(2,3,2)
histogram(stdr,10)
title('Standardized Residuals')
subplot(2,3,3)
autocorr(stdr)
subplot(2,3,4)
parcorr(stdr)
subplot(2,3,5)
qqplot(stdr)
% Durbin-Watson statistic is a commonly used test statistic in econometrics for testing autocorrelation
diffRes0 = diff(res);  
SSE0 = res'*res;
DW0 = (diffRes0'*diffRes0)/SSE0 % Durbin-Watson statistic, values close to 2 indicate no first-order autocorrelation in residuals
%% 5. Forecasting
if ~isempty(strfind(version,'2018'))||~isempty(strfind(version,'2017'))||~isempty(strfind(version,'2016'))
    [forData,YMSE] = forecast(EstMdl,step,'Y0',data);   % For MATLAB 2018 and earlier versions, use Predict_Y = forecast(EstMdl,step,'Y0',Y); For MATLAB 2019, use Predict_Y = forecast(EstMdl,step,Y);
elseif ~isempty(strfind(version,'2019'))||~isempty(strfind(version,'2020'))
    [forData,YMSE] = forecast(EstMdl,step,data);   % For MATLAB 2018 and earlier versions, use Predict_Y = forecast(EstMdl,step,'Y0',Y); For MATLAB 2019, use Predict_Y = forecast(EstMdl,step,Y);
else
    warndlg('Only supports MATLAB2016/2017/2018/2019')
end
lower = forData - 1.96*sqrt(YMSE); % 95% confidence interval lower bound
upper = forData + 1.96*sqrt(YMSE); % 95% confidence interval upper bound

figure('Visible',figflag)
plot(data,'Color',[.7,.7,.7]);
hold on
h1 = plot(length(data):length(data)+step,[data(end);lower],'r:','LineWidth',2);
plot(length(data):length(data)+step,[data(end);upper],'r:','LineWidth',2)
h2 = plot(length(data):length(data)+step,[data(end);forData],'k','LineWidth',2);
legend([h1 h2],'95% Confidence Interval','Forecast Value',...
	     'Location','NorthWest')
title('Forecast')
hold off

end

function stat = getStatAdfKpss(data)
try 
    stat = adftest(data) && ~kpsstest(data);
catch ME
    msgtext = ME.message;
    if (strcmp(ME.identifier,'econ:adftest:EffectiveSampleSizeLessThanTabulatedValues'))
         msgtext = [msgtext,'  ','Unit root test cannot be performed, data length is insufficient'];
    end
    msgbox(msgtext, 'Error')
end
end