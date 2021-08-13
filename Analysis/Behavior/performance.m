function performance(bin)
% PERFORMANCE Quick glance at behavior during session.
%   Function takes either 0 or 1 as arg.
%   0 for latest session performance overview; 1 to select a session.
%   Returns session performance overview.
%   See also GETLATESTFILE, VLINE.

%% File selection
% Automatic selection of most recent .csv file in directory
if bin == 0
    file_name = getLatestFile('C:\vr\vroutput\*.csv');
    path = 'C:\vr\vroutput\';
    config_file = 'C:\vr\vrinput\session_config.csv';
    cfid = fopen(config_file,'r');
    conf = textscan(cfid,'%s');

    [~,sess_config_name,~] = fileparts(char(conf{1}));
    sess_config = -1;
end

% Pop-up box for .csv file selection
if bin == 1
    [file_name, path] = uigetfile('*.csv', 'Select raw data', 'C:\vr\vroutput\');
    if isequal(file_name, 0)
        error('File selection was cancelled.')
    end

    [sess_config_name,~] = uigetfile('*.csv', 'Select config file', 'C:\vr\vrinput\');
    if isequal(sess_config_name, 0)
        error('File selection was cancelled.')
    end
    [~,sess_config_name,~] = fileparts(sess_config_name);
end

% Determines which config file was used for session
switch sess_config_name
    case 'MTH3_stage2'
        sess_config = 1;
    case 'MTH3_stage5'
        sess_config = 2;
    case 'MTH3_s5r'
        sess_config = 2;
    case 'MTH3_s5r_gm'
        sess_config = 2;
    case 'MTH3_VD_config'
        sess_config = 6;
    case 'MTH3_VD'
        sess_config = 6;
end

% Arg can only be 0 or 1
if bin > 1 || bin < 0
    error('Arg can only be 0 or 1.')
end

file_path = fullfile(path, file_name);

%% Load .csv with session data
data = dlmread(file_path);

% time = data(:,1);             % Time
position = data(:,2);           % Position
% t_frame = data(:,3);          % Time since last frame
speed = data(:,4);              % Running speed
vr_world = data(:,5);           % Current trial (short, long, blackbox)
valve_status = data(:,6);       % Valve status
num_trials = data(:,7);         % Number of trials
licks = data(:,8);              % Number of licks
% wheel_vel = data(:,9);        % Wheel velocity

%% Setup for quick n' dirty plots
if sess_config == 1 || sess_config == 2
    % Data on short tracks
    short_track = vr_world == 3;
    short_position = position(short_track,:);
    short_trial_num = floor(num_trials(short_track,:)/2);
    short_licks = find(licks(short_track));
    total_short_trials = length(unique(short_trial_num));
    landmark_location = 200;

    % Data on long tracks
    long_track = vr_world == 4;
    long_position = position(long_track,:);
    long_trial_num = floor(num_trials(long_track,:)/2);
    long_licks = find(licks(long_track,:));
    total_long_trials = length(unique(long_trial_num));
    landmark_location = 200;

    % Total completed trials: most times, the session stops while the animal is in
    % the middle of a trial.
    if sess_config == 1
        total_trials = total_short_trials - 1;
    elseif sess_config == 2
        total_trials = (total_short_trials + total_long_trials) - 1;
    end

end

if sess_config == 6
    %Data on shrot tracks
    short_track = vr_world == 7;
    short_position = position(short_track,:);
    short_trial_num = floor(num_trials(short_track,:)/2);
    short_licks = fidn(licks(short));
    total_short_trials = length(unique(num_trials_short));
    landmark_location = 200;

    % Data on long tracks
    long_track = vr_world == 9;
    long_position = position(long_track,:);
    long_trial_num = floor(num_trials(long_track,:)/2);
    long_licks = find(licks(long_track,:));
    total_long_trials = length(unique(long_trial_num));
    landmark_location = 200;

    % Total trials
    total_trials = (total_short_trials + total_long_trials) - 1;

end

if sess_config == -1
    error(['Config ' sess_config_name ' has not been added to the function. Please update sess_config.'])
end

% let ylim be roughly equal to the number of trials
y = total_trials + 2;

%% Reward: Triggered vs. Default @ Madison Square Garden
% Only valve changes (0 to 1 OR 0 to 2) will be considered
valve_change = diff(valve_status);
triggered_counter = 0;      % Initialize counter for triggered-to-default retaio
default_counter = 0;

temp_triggered = zeros(length(valve_change),1);
for i = 1:length(valve_change)
    if valve_change(i) == 1
        temp_triggered(i) = valve_change(i);
        triggered_counter = triggered_counter + 1;
    end
end

temp_default = zeros(length(valve_change),1);
for i = 1:length(valve_change)
    if valve_change(i) == 2
        temp_default(i-1) = 1;
        default_counter = default_counter + 1;
    end
end

temp_triggered = logical([temp_triggered;0]);
temp_default = logical([temp_default;0]);

% save indexes for easier access later
triggered_short_index = find(temp_triggered(short_track));
triggered_long_index = find(temp_triggered(long_track));
default_short_index = find(temp_default(short_track));
default_long_index = find(temp_default(long_track));

% triggered to default ratio
triggered_to_default = round(triggered_counter/default_counter,2);

% licks per reward
licks_per_reward = round((sum(licks)/(triggered_counter + default_counter)),2);

%% Average speed of the mouse
temp_speed = zeros(length(speed),1);
thresh = 0.7;                       % Minimum speed threshold

for i = 1:length(speed)-1
    if speed(i) > thresh
        temp_speed(i) = speed(i);     % Speeds under threshold will remain as 0
    end
end

speeds = temp_speed(temp_speed ~= 0);   % Holds speeds over thresholds
average_speed = round(sum(speeds)/length(speeds),2);

%% Set up data for plotting
% Licks throughout the track
short_position_licks = short_position(short_licks);              % Location of lick
short_trial_licks = short_trial_num(short_licks);                % Trial number
long_position_licks = long_position(long_licks);
long_trial_licks = long_trial_num(long_licks);

% Triggered rewards
short_position_triggered = short_position(triggered_short_index); % Triggered reward location
short_trial_triggered = short_trial_num(triggered_short_index);   % Trial number
long_position_triggered = long_position(triggered_long_index);
long_trial_triggered = long_trial_num(triggered_long_index);

% Default rewards
short_position_default = short_position(default_short_index);      % Default reward location
short_trial_default = short_trial_num(default_short_index);        % Trial number
long_position_default = long_position(default_long_index);
long_trial_default = long_trial_num(default_long_index);

%% First lick per trial
% To calculate the first lick, we throw out any licks that occurred before
% the landmark. A well-trained animal will not lick significantly before this.

short_trial_position_matrix = [short_position_licks(short_position_licks >= landmark_location), ...
    short_trial_licks(short_position_licks >= landmark_location)];
[~,is,~] = unique(short_trial_licks(short_position_licks >= landmark_location));
short_first_licks = short_trial_position_matrix(is,:);

long_trial_position_matrix = [long_position_licks(long_position_licks >= landmark_location), ...
    long_trial_licks(long_position_licks >= landmark_location)];
[~,il,~] = unique(long_trial_licks(long_position_licks >= landmark_location));
long_first_licks = long_trial_position_matrix(il,:);

try
    if max(short_first_licks(:,1)) ~= 0
        short_average_first_lick = round(mean(short_first_licks(:,1)),2);
    else
        short_average_first_lick = 0;
    end
catch
   short_average_first_lick = 0;
end

try
    if max(long_first_licks(:,1)) ~= 0
        long_average_first_lick = round(mean(long_first_licks(:,1)));
    else
        long_average_first_lick = 0;
    end
catch
    long_average_first_lick = 0;
end

if long_average_first_lick == 0
    average_first_licks = short_average_first_lick;
else
    average_first_licks = round((short_average_first_lick + long_average_first_lick)/2,2);
end

if ~isempty(long_track)
    median_first_licks = round(median(short_first_licks(:,1),'omitnan'),2);
else
    median_first_licks = round(median([short_first_licks(:,1); long_first_licks(:,1)], 'omitnan'),2);
end

%% Figure setup
figure('Name',file_name)

    % Short trials
    subplot(10,5,[2 28])

    xlabel('Location (cm)')
    xlim([50 360])
    ylim([0 y])
    ylabel('Trial #')
    title('Short trials')

    % Adds reference lines for landmark
    % vline([200 240], {'k', 'k'})
    annotation('rectangle',[.432 .445 .036 .48],'FaceColor','black','FaceAlpha',.1)

    % Adds reference lines for reward zone
    % vline([320 340], {'k', 'k'})
    annotation('rectangle', [.542 .445 .019 .48],'FaceColor', 'blue','FaceAlpha',.1)

    % Plot licks on short track
    line(short_position_licks,short_trial_licks, 'LineStyle','none','Marker','o','MarkerEdgeColor','b');

    line(short_position_triggered,short_trial_triggered,'LineStyle','none','Marker','*','MarkerEdgeColor','g');

    line(short_position_default,short_trial_default,'LineStyle','none','Marker','*','MarkerEdgeColor','r');

    % Long trials
    subplot(10,5,[4 30])

    xlabel('Location (cm)')
    xlim([50 420])
    ylim([0 y])
    title('Long trials')

    % Adds reference lines for landmark
    % vline([200 240], {'k', 'k'})
    annotation('rectangle',[.734 .445 .0325 .48],'FaceColor','black','FaceAlpha',.1)

    % Adds reference lines for reward zone box
    % vline([380 400], {'k', 'k'})
    annotation('rectangle', [.874 .445 .016 .48],'FaceColor','magenta','FaceAlpha',.1)

    % Plot licks on long track
    line(long_position_licks,long_trial_licks,long_trial_licks, 'LineStyle','none','Marker','o','MarkerEdgeColor','b');

    line(long_position_triggered,long_trial_triggered,'LineStyle','none','Marker','*','MarkerEdgeColor','g');

    line(long_position_default,long_trial_default,'LineStyle','none','Marker','*','MarkerEdgeColor','r');

    % First licks
    subplot(10,5,[37 50])

    xlabel('Location (cm)')
    xlim([190 420])
    ylim([0 y])
    title('First licks')

    % vline([200 240], {'k', 'k'})
    annotation('rectangle', [.32 .11 .1052 .227], 'FaceColor','black','FaceAlpha',.1)

    % vline([320 340], {'k', 'k'})
    annotation('rectangle', [.6393 .11 .0522 .227], 'FaceColor','blue','FaceAlpha',.1)

    % vline([380 400], {'k', 'k'})
    annotation('rectangle', [.799 .11 .0522 .227], 'FaceColor','magenta','FaceAlpha',.1)

    % Short track's first licks
    line(short_first_licks(:,1),short_first_licks(:,2),'LineStyle','none','Marker','o','MarkerEdgeColor', 'b','DisplayName', 'Short track');

    % Long track's first licks
    line(long_first_licks(:,1),long_first_licks(:,2),'LineStyle','none','Marker','o','MarkerEdgeColor','r','DisplayName','Long track');

    legend('show')

    vline(short_average_first_lick, 'b', 'Mean')
    vline(long_average_first_lick, 'r', 'Mean')
    hold on;

    % Adds textbox for session information
    axes('Position', [0.02 0.07 1 1], 'Visible', 'off');
    descr = {'Total completed trials: '
        num2str(total_trials);
        '';
        'Short trials:'
        num2str(total_short_trials);
        '';
        'Long trials:'
        num2str(total_long_trials);
        '';
        'Average velocity:'
        strcat(num2str(average_speed), ' cm/s');
        '';
        'Mean first lick:'
        num2str(average_first_licks);
        '';
        'Median first lick:'
        num2str(median_first_licks);
        ''
        'Triggered to default ratio (T/D):'
        num2str(triggered_to_default);
        ''
        'Accuracy (Licks/Rewards)'
        num2str(licks_per_reward);};
    text(0.03,0.5,descr,'FontSize', 12,'FontName','Helvetica')

    % Figure parameters
    scrsz = get(groot,'screensize');
    x0 = scrsz(3)/10;       % x position on screen
    y0 = scrsz(4)/12;       % y Position on screen
    width = 900;
    height = 600;
    set(gcf, 'units','points','position',[x0,y0,width,height])

%% Save figure
file_name = strrep(file_name, '.csv', '');
save_session = [file_name '.png'];
saveas(gcf, save_session);

end
