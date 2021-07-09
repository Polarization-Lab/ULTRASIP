%% Written by Clarissa M. DeLeon - June 26th, 2021
%% ULTRASIP_Tripod_Control
%% Code to read/write to tripod 
%define tripod class
classdef ULTRASIP_Tripod_Control < ASCOMDevice
   
    properties ( SetAccess = protected, GetAccess = public )
        
        % ASCOM Telescope properties
        tripod = alpaca_device_name; %!!! set device name !!!
    end
    
    % Read/Write
    properties ( Dependent = true, SetAccess = public, GetAccess = public )
        
        % ASCOM Telescope properties-- put which ever ones are needed here
        declination_rate; site_elevation;  site_latitude;
        site_longitude;   slew_settle_time; target_declination;
        target_right_ascension; utc_date; alignment_mode; altitude; at_home;
        at_park; azimuth; can_find_home; can_park; slewing;
        
    end
end

%MAY NEED TO PUT CLASS DEF IN ITS OWN FILE^
%Send some functions 
%-----------------------------------------------------
% Slew_To_Alt_Az
%-----------------------------------------------------
  % Slew to celestial north  
    alt = 90; 
    az = 0; 
    obj = tripod;
        function obj = Slew_To_Alt_Az( obj, alt, az )
            obj.Alpaca_Set( 'slewtoaltaz', [ 'Azimuth=' num2str( alt, '%0.9f' ) '&Altitude=' num2str( az, '%0.9f' ) ], 45 );
        end
%-----------------------------------------------------
%GET.azimuth read off current azimuth
%-----------------------------------------------------
        function value = get.azimuth( obj )
            value = obj.Alpaca_Get(  'azimuth' );
        end
%-----------------------------------------------------
% GET.altitude read off current altitude
%-----------------------------------------------------
        function value = get.altitude( obj )
            value = obj.Alpaca_Get(  'altitude' );
        end