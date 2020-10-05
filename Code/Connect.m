function success = Connect()
% CONNECT Connects to an ESP301 device if present.
% James Heath heathjam@email.arizona.edu
% Sept 25 2020
% Opens a serial connection to an ESP device and initializes it to default
% acceleration, deceleration, velocity, etc.
%
% Usage:
% success = Connect();
%
% success is a boolean that is true when the connection is successfully
% created and false when not.


	global ESP;
	success = true;
%%
	Disconnect(); % Does nothing if there's already no connection
%%
	comPort = 6; % No reason to change this on Windows. You'd need another way to do it on Mac.

	ESP = serial(sprintf('COM%0.0f', comPort));
	ESP.baudrate = 921600;
	ESP.terminator = 13; % \n
	ESP.Timeout = 1; % It replies rather quickly.
%%
	try
		fopen(ESP);
	catch
		success = false;
		ESP = [];
		return;
    end
    %%
    
    Send('VE')

	if ~Send('1AU5000') % Set max accel/decel
		success = false;
		return;
	end
	Send('2AU5000');
	Send('3AU5000');
		
	Send('1MO; 2MO; 3MO'); % Turn on motors
	
	global CURRENT_POS;
	response = Query('1TP?;2TP?;3TP?', false);
    
	if ~isempty(response) && length(response) == 3
		CURRENT_POS = response;
	else
		success = false;
		return;
	end
	
	global CURRENT_SPEED;
	if isempty(CURRENT_SPEED)
		CURRENT_SPEED = 50;
	end

	global CURRENT_ACCEL;
	if isempty(CURRENT_ACCEL)
		CURRENT_ACCEL = 50;
	end

	global CURRENT_DECEL;
	if isempty(CURRENT_DECEL)
		CURRENT_DECEL = 50;
	end
	
	for n = 1:3
		Send(sprintf('%dVA%0.5f', n, CURRENT_SPEED)); % Set axis 3 velocity
		Send(sprintf('%dAC%0.5f', n, CURRENT_ACCEL)); % Set axis 3 accel
		Send(sprintf('%dAG%0.5f', n, CURRENT_DECEL)); % Set axis 3 decel
	end
end