% Load your existing .mat file
data = load('CruiseCTD_MasterData_fromRUBoardwalk.mat');

% Get all the cruise field names
cruiseNames = fieldnames(data);

% Loop over each cruise
for iCruise = 1:length(cruiseNames)
    cruiseName = cruiseNames{iCruise};
    cruiseData = data.(cruiseName);
    
    % Get all event names for this cruise
    eventNames = fieldnames(cruiseData);
    
    % Loop over each event
    for iEvent = 1:length(eventNames)
        eventName = eventNames{iEvent};
        eventData = cruiseData.(eventName);
        
        % Check if 'DateTime' field exists and convert it
        if isfield(eventData, 'DateTime')
            % Convert datetime to string format
            eventData.DateTime = datestr(eventData.DateTime, 'yyyy-mm-dd HH:MM:SS');
        end
        
        % Assign the modified event data back to the structure
        cruiseData.(eventName) = eventData;
    end
    
    % Assign the modified cruise data back to the main structure
    data.(cruiseName) = cruiseData;
end

% Save the modified data to a new .mat file
save('CruiseCTD_MasterData_fromRUBoardwalk_DateAdjusted.mat', '-struct', 'data');
