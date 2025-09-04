classdef reservoir < PRORES
    methods
        %% Default settings of the problem
        
        function Setting(obj)

            % Dead water level
            obj.lower    = [963.11, 952, 952, 952, 945, 945, 945, 945, 945, 945, 952, 952, 820.38, 790, 790, 790, 765, 765, 765, 765, 765, 765, 790, 790, 589.82, 560, 560, 560, 540, 540, 540, 540, 540, 540, 560, 560, 375.448, 374.5, 374.5, 374.5, 370, 370, 370, 370, 370, 370, 374.5, 374.5];
            %obj.lower    = [975, 970, 965, 960, 955, 950, 945, 945, 950, 955, 960, 965, 825, 815, 805, 795, 785, 775, 765, 765, 775, 785, 795, 805, 600, 590, 580, 570, 560, 550, 540, 540, 550, 560, 570, 580, 380, 378.5, 377, 375.5, 374, 372.5, 370, 370, 372.5, 374, 375.5, 377];
           %obj.lower    = [975, 945, 945, 945, 952, 952, 945, 945, 952, 952, 945, 945, 825, 765, 765, 765, 765, 765, 765, 765, 765, 765, 765, 765, 600, 540, 540, 540, 540, 540, 540, 540, 540, 540, 540, 540, 380, 378.5, 377, 375.5, 374, 372.5, 370, 370, 372.5, 374, 375.5, 377];
            % Normal water storage level
            obj.upper    = [963.11, 975, 975, 975, 975, 975, 952, 952, 975, 975, 975, 975, 820.38, 825, 825, 825, 825, 825, 790.5, 790.5, 825, 825, 825, 825, 589.82, 600, 600, 600, 600, 600, 560, 560, 600, 600, 600, 600, 375.448, 380, 380, 380, 380, 380, 374.5, 374.5, 380, 380, 380, 380];
            %obj.upper    = [963.11, 975, 975, 975, 975, 952, 952, 952, 952, 975, 975, 975, 820.38, 825, 825, 825, 825, 790, 790, 790, 790, 825, 825, 825, 589.82, 600, 600, 600, 600, 560, 560, 560, 560, 600, 600, 600, 375.448, 380, 380, 380, 380, 380, 374.5, 374.5, 374.5, 380, 380, 380];
            %obj.upper    = [975, 975, 975, 975, 975, 975, 952, 952, 975, 975, 975, 975, 825, 825, 825, 825, 825, 825, 790, 790, 825, 825, 825, 825, 600, 600, 600, 600, 600, 600, 560, 560, 600, 600, 600, 600, 380, 380, 380, 380, 380, 380, 374.5, 374.5, 380, 380, 380, 380];

            %             obj.upper    = [963.11, 975, 975, 975, 975, 975, 975, 975, 975, 975, 975, 975, 820.38, 825, 825, 825, 825, 825, 825, 825, 825, 825, 825, 825, 589.82, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 375.448, 380, 380, 380, 380, 380, 380, 380, 380, 380, 380, 380];
            % Encoding format, real number encoding here
            obj.encoding = ones(1,obj.D);
            % Initial water level
            obj.InitialLevel = [963.11, 820.38, 589.82, 375.448];
            % Cascade terminal water level
            obj.SanxiaLevel = [267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267];
            % Minimum power output
            obj.Min_Capcity = [0, 0, 0, 0];
            % Maximum power output
            obj.Max_Capcity = [10200, 16000, 12600, 6000];
            % Minimum discharge flow
            obj.Low_Discharge = [900, 1260, 1200, 1200];
            % Maximum discharge flow
            obj.High_Discharge = [35800, 38800, 40888, 41200];
            % Inflow
            % Directly read WDD data
       wdddata = readtable('Runoff-Wudongde-Monthly-Data-Standardized.xlsx');
       year_col = wdddata.Var1;
       row = find(year_col == 1994);
       if ~isempty(row)
          obj.Input = table2array(wdddata(row,2:13)); % Columns 2-13 for 12 months
       else
        error('Data not found');
       end
            % Days per month
            obj.M_d = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        end
        %% Calculate objective values, PopObj stores three objectives with dimensions i rows and 3 columns, PopDec is water level
        function [PopObj,PowerEnergy] = CalObj(obj,PopDec,PopOutput)
        % Define Vdown (lower limit storage capacity, unit: 100 million cubic meters)  
        Vdown = [34, 34, 34, 34, 29, 29, 29, 29, 29, 29, 34, 34, ... % 1st reservoir
         118, 118, 118, 118, 86, 86, 86, 86, 86, 86, 118, 118, ... % 2nd reservoir
         70, 70, 70, 70, 54, 54, 54, 54, 54, 54, 70, 70, ... % 3rd reservoir
         44, 44, 44, 44, 40, 40, 40, 40, 40, 40, 44, 44];
        % Define Vup (upper limit storage capacity, unit: 100 million cubic meters)
        Vup = [57, 57, 57, 57, 57, 57, 35, 35, 57, 57, 57, 57, ... % 1st reservoir
        183, 183, 183, 183, 183, 183, 120, 120, 183, 183, 183, 183, ... % 2nd reservoir
        116, 116, 116, 116, 116, 116, 71, 71, 116, 116, 116, 116, ... % 3rd reservoir
        50, 50, 50, 50, 50, 50, 45, 45, 50, 50, 50, 50]; % 4th reservoir
            Two_dif = zeros(size(PopDec));
            for i = 1:size(PopDec,1)
                % First three reservoirs
                for j = 1:(obj.ResNum-1)*obj.ResMonthNum
                    if mod(j,obj.ResMonthNum)==0
                        Two_dif(i,j)=(PopDec(i,j)+PopDec(i,j-obj.ResMonthNum+1))/2-PopDec(i,j+obj.ResMonthNum);
                    else
                        Two_dif(i,j)=(PopDec(i,j)+(PopDec(i,j+1)))/2-PopDec(i,j+obj.ResMonthNum);
                    end
                end
                % The fourth reservoir is related to Three Gorges water level
                for j = 1:obj.ResMonthNum
                    extra = obj.ResMonthNum*3+j;
                    if mod(j,obj.ResMonthNum)==0
                        Two_dif(i,extra)=(PopDec(i,extra)+PopDec(i,extra-obj.ResMonthNum+1))/2-obj.SanxiaLevel(1,j);
                    else
                        Two_dif(i,extra)=(PopDec(i,extra)+PopDec(i,extra+1))/2-obj.SanxiaLevel(1,j);
                    end
                end
            end
            % Power generation
            C = 8; % Power coefficient
            PopObj = zeros(obj.N,obj.M);
            for i = 1:size(PopDec,1)
                for j = 1:obj.D
                    PowerEnergy(i,j) = C*Two_dif(i,j)*PopOutput(i,j)/1000;
                    if PowerEnergy(i,j) >= obj.Max_Capcity(1,floor((j-1)/12)+1)
                        PowerEnergy(i,j) = obj.Max_Capcity(1,floor((j-1)/12)+1);
                    end
                    New_M_d = repmat(obj.M_d,1,obj.ResNum);
                    PopObj(i,1) = PopObj(i,1) + PowerEnergy(i,j)*1000*New_M_d(1,j)*24/(10^8);
                end
            end
            % Flood control (reservoir safety, considering May-June and September-October)
            S_c = obj.CalCapcity(PopDec);
            flood_indices = [5,6,9,10,17, 18, 21,22 ,29,30, 33, 34,41,42, 45, 46];
            for i = 1:size(PopDec,1)
                normalized_sum=0;
                for j = flood_indices 
            % Extract current storage capacity value
            current_S_c = S_c(i,j);          
            % 
            Vmin = Vdown(j);
            Vmax = Vup(j);
            
             % Add division by zero protection
             if (Vmax - Vmin) ~= 0
                 normalized_value = (Vmax - current_S_c) / (Vmax - Vmin);
             else
                 normalized_value = 0;  % Set to 0 when upper and lower limits are equal
             end
             normalized_sum = normalized_sum + normalized_value;
                end
                
                supply_indices = [1,2,3,4,11,12, 13,14,15,16, 23, 24,25,26, 27,28,35,36,37,38, 39,40, 47,48];
           
                supply_normalized_sum=0;
                for j = supply_indices 
                    
            % Extract current storage capacity value
            supply_current_S_c = S_c(i,j);
             Vmin2 = Vdown(j);
            Vmax2 = Vup(j);
            
             % Add division by zero protection
             if (Vmax2 - Vmin2) ~= 0
                 supply_normalized_value = (supply_current_S_c - Vmin2) / (Vmax2 - Vmin2);
             else
                 supply_normalized_value = 0;  % Set to 0 when upper and lower limits are equal
             end
             supply_normalized_sum = supply_normalized_sum + supply_normalized_value;
                end 
         % Downstream flood risk
         floodcul7 = (PopOutput(i,7)+PopOutput(i,19)+PopOutput(i,31)+PopOutput(i,43))/(35800+38800+40888+41200);
         nfloodcul7=1- floodcul7;
         floodcul8 = (PopOutput(i,8)+PopOutput(i,20)+PopOutput(i,32)+PopOutput(i,44))/(35800+38800+40888+41200);
         nfloodcul8=1- floodcul8;
         % Store cumulative sum in objective 2
            PopObj(i,2) =  ((normalized_sum / 16)+(supply_normalized_sum / 24)+min(nfloodcul7,nfloodcul8))/3;
            end
           
            for i = 1:size(PopDec,1)
               
            Drop_area_i =obj.CalFluctuationZoneArea(PopDec(i,:));
            selected_cols = [5,6,9,10,17,18,21,22,29,30,33,34,41,42,45,46];
            Total_drop_area_sum = sum(Drop_area_i(:, selected_cols), 2);
            
            % Calculate CO2 emissions (unit: 10,000 tons)
            CO2_C=Total_drop_area_sum*1821*30*3.6644;
            % Add division by zero protection when calculating sed1
            denominator = (obj.Input(1, 7)+obj.Input(1, 8))/2 * 0.30175*62*24*3600;
            if denominator == 0
                sed1 = 0;  % Set sed1 to 0 when denominator is 0
            else
                sed1 = (((PopOutput(i, 43)+PopOutput(i, 44))/2 * 0.03088*62*24*3600) / denominator);
            end
              
            % Add division by zero protection to prevent division by zero error when min function returns 0
            min_output_1 = min(PopOutput(i, [1:4, 11:12]));
            min_output_2 = min(PopOutput(i, [13:16, 23:24]));
            min_output_3 = min(PopOutput(i, [25:28, 35:36]));
            min_output_4 = min(PopOutput(i, [37:40, 47:48]));
            
            % Prevent division by zero, set to a small positive number when minimum value is 0
            if min_output_1 == 0, min_output_1 = 1e-10; end
            if min_output_2 == 0, min_output_2 = 1e-10; end
            if min_output_3 == 0, min_output_3 = 1e-10; end
            if min_output_4 == 0, min_output_4 = 1e-10; end
            
            PopObj(i,3) = ((1-((CO2_C/10000000000000)/(184988037821760/10000000000000))) + ...
                          (sed1) + ...
                          (((1-900/min_output_1) + (1-1260/min_output_2) + ...
                            (1-1200/min_output_3) + (1-1200/min_output_4))/4))/3;
            end
     

        end

 %% Calculate fluctuation zone area
 function Drop_area = CalFluctuationZoneArea(obj, PopDec)
    % Calculate reservoir fluctuation zone area (lower water level results in larger fluctuation zone area)
    % Input: PopDec (N×48), every 12 columns correspond to water level data of one reservoir (unit: meters)
    % Output: FluctuationZone_area (N×48), fluctuation zone area (unit: square meters)
    
    [N, ~] = size(PopDec);
    Drop_area = zeros(size(PopDec));
    
    % Define water level-area interpolation points for each reservoir's fluctuation zone (note reversed order)
    % Format: {[water level range], [maximum fluctuation zone area (at lowest water level), 0 (at highest water level)]}
    reservoir_data = {
        {[955, 975], [56290000, 0]};     % Wudongde: 56290000m² at 955m, 0m² at 975m
        {[765, 825], [96620000, 0]};     % Baihetan
        {[540, 600], [65600000, 0]};     % Xiluodu
        {[370, 380], [12510000, 0]};      % Xiangjiaba
    };
    
    % Batch process 12-month data for each reservoir
    for reservoir_idx = 1:4
        start_col = (reservoir_idx-1)*12 + 1;
        end_col = reservoir_idx*12;
        water_levels = PopDec(:, start_col:end_col);  % Extract current reservoir's water level data
        
        % Get interpolation data (water level ascending, area descending)
        levels = reservoir_data{reservoir_idx}{1};
        area_range = reservoir_data{reservoir_idx}{2};
        
        % Linear interpolation to calculate fluctuation zone area
        Drop_area(:, start_col:end_col) = ...
            interp1(levels, area_range, water_levels, 'linear', 'extrap');
    end
    
    % Ensure area is non-negative and does not exceed theoretical maximum
    Drop_area = max(Drop_area, 0);
    for reservoir_idx = 1:4
        max_area = reservoir_data{reservoir_idx}{2}(1);  % Get current reservoir's maximum fluctuation zone area
        start_col = (reservoir_idx-1)*12 + 1;
        end_col = reservoir_idx*12;
        Drop_area(:, start_col:end_col) = ...
            min(Drop_area(:, start_col:end_col), max_area);
    end
end
 %% Calculate storage capacity and outflow

        % Water level to storage capacity conversion for each reservoir
        function S_c = CalCapcity(obj,PopDec)
%             for i = 1:obj.N
            S_c = zeros(size(PopDec));
            for i = 1:size(PopDec,1)  
                % Wudongde water level to storage capacity conversion
                for j = 1:obj.ResMonthNum
                    S_c(i, j) = 1.00e-62 * PopDec(i,j)^21.33;    
                end
                % Baihetan water level to storage capacity conversion
                for j = obj.ResMonthNum+1:2*obj.ResMonthNum
                   S_c(i, j) = 1.53e-27 * PopDec(i,j)^9.97;
                end
                % Xiluodu water level to storage capacity conversion
                for j = 2*obj.ResMonthNum+1:3*obj.ResMonthNum
                   S_c(i, j) = 3.40e-18 * PopDec(i,j)^7.03;
                end
                % Xiangjiaba water level to storage capacity conversion
                for j = 3*obj.ResMonthNum+1:4*obj.ResMonthNum
                    S_c(i, j) = 3.55e-18 * PopDec(i,j)^7.42;
                end
              
            end
        end
        % Calculate outflow for each reservoir
        function Output = CalOutput(obj,S_c)
            Output = zeros(obj.N,obj.D);
%             for i = 1:obj.N
            for i = 1:size(S_c,1)
                % Wudongde outflow                
                for j = 1:obj.ResMonthNum-1
                    Output(i, j) = obj.Input(1, j) - (S_c(i, j + 1) - S_c(i, j)) * (10 ^ 8) / (obj.M_d(1, j) * 24 * 3600);
                end
                Output(i, obj.ResMonthNum) = obj.Input(1, obj.ResMonthNum) - (S_c(i, 1) - S_c(i, 12)) * (10 ^ 8) / (obj.M_d(1, obj.ResMonthNum) * 24 * 3600);
                % Baihetan outflow
                for j = obj.ResMonthNum+1:2*obj.ResMonthNum-1
                    Output(i, j) = Output(i, j-obj.ResMonthNum) - (S_c(i, j + 1) - S_c(i, j)) * (10 ^ 8) / (obj.M_d(1, j-obj.ResMonthNum) * 24 * 3600);
                end
                Output(i, 2*obj.ResMonthNum) = Output(i, obj.ResMonthNum) - (S_c(i, obj.ResMonthNum+1) - S_c(i, obj.ResMonthNum+12)) * (10 ^ 8) / (obj.M_d(1, obj.ResMonthNum) * 24 * 3600);
                % Xiluodu outflow
            
                for j = 2*obj.ResMonthNum+1:3*obj.ResMonthNum-1
                    Output(i, j) = Output(i, j-obj.ResMonthNum) - (S_c(i, j + 1) - S_c(i, j)) * (10 ^ 8) / (obj.M_d(1, j-2*obj.ResMonthNum) * 24 * 3600);
                end
                Output(i, 3*obj.ResMonthNum) = Output(i, 2*obj.ResMonthNum) - (S_c(i, 2*obj.ResMonthNum+1) - S_c(i, 2*obj.ResMonthNum+12)) * (10 ^ 8) / (obj.M_d(1, obj.ResMonthNum) * 24 * 3600);
                % Xiangjiaba outflow
                for j = 3*obj.ResMonthNum+1:4*obj.ResMonthNum-1
                    Output(i, j) = Output(i, j-obj.ResMonthNum) - (S_c(i, j + 1) - S_c(i, j)) * (10 ^ 8) / (obj.M_d(1, j-3*obj.ResMonthNum) * 24 * 3600);
                end
                Output(i, 4*obj.ResMonthNum) =  Output(i, 3*obj.ResMonthNum) - (S_c(i, 3*obj.ResMonthNum+1) - S_c(i, 3*obj.ResMonthNum+12)) * (10 ^ 8) / (obj.M_d(1, obj.ResMonthNum) * 24 * 3600);
               
            end
        end
        %% Calculate constraint violations
        function PopCon = CalCon(obj,PopDec,PopOutput,PowerEnergy)
            PopCon = zeros(obj.N,1);
            % Reservoir water level constraints
            wl = zeros(obj.N,1);
%             for i = 1:obj.N
            for i = 1:size(PopDec,1)
                for j = 1:obj.D
                    wl(i,1) = wl(i,1) + max(0,obj.lower(1,j)-PopDec(i,j))+max(0,PopDec(i,j)-obj.upper(1,j));
                end
            end
            % Reservoir discharge flow constraints
            rdf = zeros(obj.N,1);
%             for i = 1:obj.N
            for i = 1:size(PopDec,1)
                for j = 1:obj.ResNum
                    for k = 1:obj.ResMonthNum
                        rdf(i,1) = rdf(i,1) + max(0, obj.Low_Discharge(1,j)-PopOutput(i,k+(j-1)*obj.ResMonthNum))+max(0,PopOutput(i,k+(j-1)*obj.ResMonthNum)-obj.High_Discharge(1,j));
                    end
                end
            end
            % Power station output constraints
            pso = zeros(obj.N,1);
%             for i = 1:obj.N
            for i = 1:size(PopDec,1)
                for j = 1:obj.ResNum
                    for k = 1:obj.ResMonthNum
                        pso(i,1) = pso(i,1) + max(0, obj.Min_Capcity(1,j)-PowerEnergy(i,k+(j-1)*obj.ResMonthNum))+max(0,PowerEnergy(i,k+(j-1)*obj.ResMonthNum)-obj.Max_Capcity(1,j));
                    end
                end
            end
            % Total constraint violations
            PopCon = wl+rdf+pso;
        end
    end
end