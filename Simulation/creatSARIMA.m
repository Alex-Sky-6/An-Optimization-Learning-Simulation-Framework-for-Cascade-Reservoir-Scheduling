function Mdl = creatSARIMA(ar,ma,sar,sma,season,di)

    if ar==0&&ma~=0&&sar==0&&sma~=0
        Mdl = arima('Constant',0,'D',di,...
    'Seasonality',season,'MALags',1:ma,'SMALags',(1:sma)*season);
    elseif  ar~=0&&ma==0&&sar==0&&sma~=0
        Mdl = arima('Constant',0,'ARLags',1:ar,'D',di,...
    'Seasonality',season,'SMALags',(1:sma)*season);
    elseif  ar==0&&ma~=0&&sar~=0&&sma==0
        Mdl = arima('Constant',0,'SARLags',(1:sar)*season,'D',di,...
    'Seasonality',season,'MALags',1:ma);
    elseif  ar~=0&&ma==0&&sar~=0&&sma==0
        Mdl = arima('Constant',0,'ARLags',1:ar,'SARLags',(1:sar)*season,'D',di,...
    'Seasonality',season);
    elseif  ar~=0&&ma~=0&&sar~=0&&sma==0
        Mdl = arima('Constant',0,'ARLags',1:ar,'SARLags',(1:sar)*season,'D',di,...
    'Seasonality',season,'MALags',1:ma);
    elseif  ar~=0&&ma~=0&&sar==0&&sma~=0
        Mdl = arima('Constant',0,'ARLags',1:ar,'D',di,...
    'Seasonality',season,'MALags',1:ma,'SMALags',(1:sma)*season);
    elseif  ar~=0&&ma==0&&sar~=0&&sma~=0
        Mdl = arima('Constant',0,'ARLags',1:ar,'SARLags',(1:sar)*season,'D',di,...
    'Seasonality',season,'SMALags',(1:sma)*season);
    elseif  ar==0&&ma~=0&&sar~=0&&sma~=0
        Mdl = arima('Constant',0,'SARLags',(1:sar)*season,'D',di,...
    'Seasonality',season,'MALags',1:ma,'SMALags',(1:sma)*season);
    elseif  ar~=0&&ma~=0&&sar~=0&&sma~=0
        Mdl = arima('Constant',0,'ARLags',1:ar,'SARLags',(1:sar)*season,'D',di,...
    'Seasonality',season,'MALags',1:ma,'SMALags',(1:sma)*season);
    end
end