function varargout = check_rois_beh(varargin)
% CHECK_ROIS_BEH MATLAB code for check_rois_beh.fig
%      CHECK_ROIS_BEH, by itself, creates a new CHECK_ROIS_BEH or raises the existing
%      singleton*.
%
%      H = CHECK_ROIS_BEH returns the handle to a new CHECK_ROIS_BEH or the handle to
%      the existing singleton*.
%
%      CHECK_ROIS_BEH('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CHECK_ROIS_BEH.M with the given input arguments.
%
%      CHECK_ROIS_BEH('Property','Value',...) creates a new CHECK_ROIS_BEH or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before check_rois_beh_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to check_rois_beh_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help check_rois_beh

% Last Modified by GUIDE v2.5 17-Jul-2017 16:08:34

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @check_rois_beh_OpeningFcn, ...
                   'gui_OutputFcn',  @check_rois_beh_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before check_rois_beh is made visible.
function check_rois_beh_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to check_rois_beh (see VARARGIN)

% Choose default command line output for check_rois_beh
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes check_rois_beh wait for user response (see UIRESUME)
% uiwait(handles.check_rois_beh);


% --- Outputs from this function are returned to the command line.
function varargout = check_rois_beh_OutputFcn(hObject, ~, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in load.
function load_Callback(hObject, ~, handles)
% hObject    handle to load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% get imaging file name and path
try
    % if running on any mac with lab's external hard drive
    if ismac
         [handles.file_name, pathstr] = uigetfile('M01_000_*_rigid.*', 'Select imaging data',...
             '/Volumes/LaCie');
    end

    % if running on lab's pcs
    % NOTE: for ease of use, change path if file structure is altered in the future
    if ispc
        [handles.file_name, pathstr] = uigetfile('M01_000_*_rigid.*', 'Select imaging data',...
            'F:\');
    end
catch
    error('There was a problem retrieving your file.')
end

assert(~isequal(handles.file_name,0), 'Imaging file selection cancelled.')

% get behavioral data name and path
try
    if ismac
        [handles.beh,path_beh] = uigetfile('MTH3_vr1_*.csv', 'Select behavioral data',...
            '/Volumes/LaCie');
    end
    if ispc
        [handles.beh, path_beh] = uigetfile('MTH3_vr1_*.csv', 'Select behavioral data',...
            'F:\');
    end
catch
    error('There was a problem retrieving your file.')
end

assert(~isequal(handles.beh,0), 'Behavioral file selection cancelled.')

file_path = fullfile(path_beh, handles.beh);
data = dlmread(file_path);

% IMAGING DATA
[~, name, ~] = fileparts(handles.file_name);

handles.file_name = [pathstr name];

disp('File selected:')
disp(['     ' name])

% Set up extensions for analysis
sig_file = [handles.file_name '.sig'];
sbx_file = [handles.file_name '.sbx'];

% open .sig file (for plot)
handles.roi_data = dlmread(sig_file); 

% initialize necessary handles for plotting/general reference
handles.col = 1;
handles.tmp = ones(1, size(handles.roi_data,2));
handles.comment = strings(1, size(handles.roi_data,2));

% plot roi graph
handles.graph = plot(handles.roi_plot, handles.roi_data(:,handles.col));
xlim([0 27800])

% open .sbx file (for mean roi image)
[path, file_name, ~] = fileparts(sbx_file);
sbx = strcat(path, filesep, file_name);

% process image once for future access
visual = sbxmakeref(sbx, 300, 1);
handles.mean_vis = mean(visual,3);

% contains roi mask; see imagesc(handles.mask)
mask_file = load([handles.file_name '.segment'], '-mat');

% mask_file contains the structure mask
% for easier (and global) access, we set this mask to a handle
handles.mask = mask_file.mask;

% get rows and columns corresponding to the current ROI (determined by col)
[r,c] = find(handles.mask == handles.col);

% rough calculatations of center of ROI
handles.r_min = min(r);
handles.r_max = max(r);
r_center = mean(handles.r_min, handles.r_max);

if mod(r_center,2) ~= 0
    r_center = floor(mean(handles.r_min,handles.r_max));
end

handles.c_min = min(c);
handles.c_max = max(c);
c_center = mean(handles.c_min, handles.c_max);

if mod(c_center,2) ~= 0
    c_center = floor(mean(handles.c_min,handles.c_max));
end

% display mean image ROI in gui
handles.roi = imagesc(handles.roi_vis, handles.mean_vis(r_center-10:r_center+20,c_center-10:c_center+30));
set(handles.roi_vis,'YTick',[])
set(handles.roi_vis,'XTick',[])
colormap(gray)

% print current roi on top of graph
set(handles.roi_num, 'String', strcat('ROI ', num2str(handles.col)));

% BEHAVIORAL DATA
% set up time and position
t = data(:,1);
pos = data(:,2);
vr_world = data(:,5);   % keep track of blackboxes

corridor = logical(vr_world ~= 5);

for i = 1:length(corridor)
    if corridor(i) == 0
        pos(i) = 0;
    end
end

ax(handles.beh_plot);
plot(handles.beh_plot, t, pos);
xlim([0 900]);

% 30 FRAMES PER SECOND
% DIVIDE FLUORESCENCE BY 30 TO GET THE TIME

guidata(hObject, handles)


% --- Executes on button press in next.
function next_Callback(hObject, ~, handles)
% hObject    handle to next (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% go to next col
handles.col = handles.col + 1;

% if user tries to go over max rois, reset to max roi
if handles.col > (size(handles.roi_data, 2)/3)
    handles.col = (size(handles.roi_data, 2)/3);
    disp('There are no more ROIs to display.')
end

% clear roi_plot axes
cla(handles.roi_plot)
handles.graph = plot(handles.roi_plot, handles.roi_data(:,handles.col));
xlim([0 27800])

% again, more calculations. boring stuff. could be optimized?
[r,c] = find(handles.mask == handles.col);
r_min = min(r);
r_max = max(r);
c_min = min(c);
c_max = max(c);

r_center = mean(r_min,r_max);
c_center = mean(c_min,c_max);

if mod(r_center,2) ~= 0
    r_center = floor(mean(r_min,r_max));
end
if mod(c_center,2) ~= 0
    c_center = floor(mean(c_min,c_max));
end

% clear roi_vis for new image
cla(handles.roi_vis)
handles.roi = imagesc(handles.roi_vis, handles.mean_vis(r_center-10:r_center+20,c_center-10:c_center+30));
set(handles.roi_vis,'YTick',[])
set(handles.roi_vis,'XTick',[])
colormap(gray)

set(handles.roi_num, 'String', strcat('ROI ', num2str(handles.col)));

guidata(hObject, handles)


% --- Executes on button press in prev.
function prev_Callback(hObject, ~, handles)
% hObject    handle to prev (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% go to previous col
handles.col = handles.col - 1;

% to evade error messages, reset to 1 if user tries to go below 0
if handles.col <= 0
    handles.col = 1;
    disp('There are no more ROIs to display')
    
end

% clear roi_plot axes for new plot
cla(handles.roi_plot)
handles.graph = plot(handles.roi_plot, handles.roi_data(:,handles.col));
xlim([0 27800])

% calculations for roi center
[r,c] = find(handles.mask == handles.col);
r_min = min(r);
r_max = max(r);
c_min = min(c);
c_max = max(c);
r_center = mean(r_min,r_max);
c_center = mean(c_min,c_max);

if mod(r_center,2) ~= 0
    r_center = floor(mean(r_min,r_max));
end
if mod(c_center,2) ~= 0
    c_center = floor(mean(c_min,c_max));
end

% clear roi_vis axes for new image
cla(handles.roi_vis)
handles.roi = imagesc(handles.roi_vis, handles.mean_vis(r_center-10:r_center+20,c_center-10:c_center+30));
set(handles.roi_vis,'YTick',[])
set(handles.roi_vis,'XTick',[])
colormap(gray)

set(handles.roi_num, 'String', strcat('ROI ', num2str(handles.col)));

guidata(hObject, handles)


function jump_to_Callback(hObject, ~, handles)
% hObject    handle to jump_to (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of jump_to as text
%        str2double(get(hObject,'String')) returns contents of jump_to as a double

% set col to be the input
handles.col = str2double(get(hObject, 'String'));

% if user inputs a col num higher than the max, warn user
assert(handles.col <= (size(handles.roi_data,2)/3), ['This session has ',...
    '%d ROIs in total. \nPlease choose another ROI.'], size(handles.roi_data, 2)/3)

% clear roi_plot axes
cla(handles.roi_plot)
handles.graph = plot(handles.roi_plot, handles.roi_data(:,handles.col));
xlim([0 27800])

% calculations, yay
[r,c] = find(handles.mask == handles.col);
r_min = min(r);
r_max = max(r);
c_min = min(c);
c_max = max(c);
r_center = mean(r_min,r_max);
c_center = mean(c_min,c_max);

if mod(r_center,2) ~= 0
    r_center = floor(mean(r_min,r_max));
end
if mod(c_center,2) ~= 0
    c_center = floor(mean(c_min,c_max));
end

% clear roi_vis axes
cla(handles.roi_vis)
handles.roi = imagesc(handles.roi_vis, handles.mean_vis(r_center-10:r_center+20,c_center-10:c_center+30));
set(handles.roi_vis,'YTick',[])
set(handles.roi_vis,'XTick',[])
colormap(gray)

set(handles.roi_num, 'String', strcat('ROI ', num2str(handles.col)));

guidata(hObject, handles)


% --- Executes on button press in reject.
function reject_Callback(hObject, ~, handles)
% hObject    handle to reject (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% a new file will be created with a logical vector, going from 1 to
% the number of rois. 0 means the roi was rejected, 1 means it wasn't.

% change tmp value at col position if ROI is rejected 
handles.tmp(1, handles.col) = 0;

guidata(hObject, handles)


function comment_input_Callback(hObject, ~, handles)
% hObject    handle to comment_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of comment_input as text
%        str2double(get(hObject,'String')) returns contents of comment_input as a double

% assign comments to their respective rois
handles.comment(1, handles.col) = strcat('(', num2str(handles.col), ')', get(hObject, 'string'), ';');

guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function comment_input_CreateFcn(hObject, ~, handles)
% hObject    handle to comment_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on slider movement.
function slider_Callback(hObject, ~, handles)
% hObject    handle to slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

handles.slider = hObject;

slider_pos = get(handles.slider, 'Value');      % Slider position

axes(handles.roi_plot)
xLims = xlim;
% set(handles.slider, 'Min', min(xLims));
% set(handles.slider, 'Max', max(xLims));
% 
set(handles.roi_plot, 'Xlim', xLims);

guidata(hObject, handles)

% --- Executes on button press in save.
function save_Callback(hObject, ~, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% clicking save will create a bunch of files

% file_name.txt will contain data of whether it was rejected or not
save_rois = [handles.file_name '.txt'];
fileID = fopen(save_rois, 'w');
fprintf(fileID, '%s', num2str(handles.tmp));
fclose(fileID);

% file_name.csv will contain comment data, along with the roi number
save_comments = [handles.file_name '.csv'];
commentsID = fopen(save_comments, 'w');
fprintf(commentsID, '%s', handles.comment);
fclose(commentsID);


% --- Executes during object creation, after setting all properties.
function roi_plot_CreateFcn(hObject, ~, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1

handles.roi_plot = hObject;
guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function roi_num_CreateFcn(hObject, ~, handles)
% hObject    handle to roi_num (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

handles.roi_num = hObject;

% --- Executes during object creation, after setting all properties.
function jump_to_CreateFcn(hObject, ~, handles)
% hObject    handle to jump_to (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function roi_vis_CreateFcn(hObject, ~, handles)
% hObject    handle to roi_vis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate roi_vis

handles.roi_vis = hObject;
set(handles.roi_vis,'YTick',[])
set(handles.roi_vis,'XTick',[])
guidata(hObject, handles)


function date_Callback(hObject, eventdata, handles)
% hObject    handle to date (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of date as text
%        str2double(get(hObject,'String')) returns contents of date as a double

handles.date = get(hObject,'String');

guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function date_CreateFcn(hObject, eventdata, handles)
% hObject    handle to date (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
