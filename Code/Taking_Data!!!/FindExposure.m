% ExposureTest -- Take picture of setup with varying exposures and plot counts as
% a function of exposure time in seconds.  NOTE: min exposure time is 30 ms
%
% Written by Atkin Hyatt 07/01/2021
% Last modified by Atkin Hyatt 07/19/2021

%% Initialize
stop(vid)
a1 = 30000; a2 = 50000;     % Linear region counts constraints
a = 45000;                  % Optimum counts
Move_motor(0, ELL14);

% Guess first point
guess = input('Guess? '); expo1 = guess;
src.ExposureTime = expo1;
if guess >= 5
   vid.Timeout = 2 * guess;
end

start(vid)
image = UV_data(vid,framesPerTrigger); % take picture
countFeedback = max(max(image));
fprintf("%f counts\n", countFeedback)

%% Find Linear Region
count = 0;
while countFeedback < a1 || countFeedback > a2 && expo1 > 0
    count = count + 1; fprintf("Test #%d\n", count);
    if countFeedback < a1
        expo1 = expo1 + guess/0.01;
    elseif countFeedback > a2
        expo1 = expo1 - guess/10;
    end
    src.ExposureTime = expo1;
    image = UV_data(vid,framesPerTrigger); % take picture
    countFeedback = max(max(image)); fprintf("%f counts\n", countFeedback)
end
if expo1 < 0
    fprintf("WAY too exposed\n")
end
avCounts1 = countFeedback;

%% Find Second Point in Linear Region

% exposure time
if guess > expo1
    expo2 = expo1 - 0.01;
else
    expo2 = expo1 + 0.01;
end
src.ExposureTime = expo2;

% counts
fprintf("\nFinding second point in linear region...\n")
image = UV_data(vid,framesPerTrigger); % take picture
avCounts2 = max(max(image));
fprintf("%f counts\n", avCounts2)

stop(vid)
%% Estimate Optimum Exposure
m = (avCounts2 - avCounts1) / (expo2 - expo1);
opEx = (a - avCounts1) / m + expo1;
fprintf("Optimum exposure is %f\n", opEx)