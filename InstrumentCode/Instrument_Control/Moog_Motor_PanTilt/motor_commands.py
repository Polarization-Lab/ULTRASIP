# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 11:26:41 2023

This code provides functions for commands to control the Moog motor

@author: C.M.DeLeon
Acknowledgement: Sierra Macleod 

"""
#Import Libraries
import serial
import time

# Define constants

# Control chars
CTRL_STX = 0x02
CTRL_ETX = 0x03
CTRL_ACK = 0x06
CTRL_NACK = 0x15
CTRL_ESC = 0x1B


# Identity - 00 will always work
IDENTITY = 0x00

# Command chars
CMD_GET_STAT_JOG = 0x31
CMD_MV_ABS = 0x35
CMD_MV_HOME = 0x36
CMD_MV_ENT_COORD = 0x33
CMD_ALIGN_ENT_COORD = 0x83
# ... etc. you probably don't need to write one for every command,
# but it's good form to make constants for the ones you plan to use.
# makes it more readable for future you and anyone else reading it

# Status bit offsets
PAN_CWSL = 7   # Clockwise soft limit reached
PAN_CCWSL = 6  # Counter-clockwise soft limit reached
PAN_CWHL = 5   # Clockwise hard limit reached
PAN_CCWHL = 4  # Counter-clockwise hard limit reached
PAN_TO = 3     # Timeout
PAN_DE = 2     # Direction error
PAN_OL = 1     # Current overload
PAN_PRF = 0    # Resolver fault

TILT_USL = 7  # Up soft limit reached
TILT_DSL = 6  # Down soft limit reached
TILT_UHL = 5  # Up hard limit reached
TILT_DHL = 4  # Down hard limit reached
TILT_TO = 3   # Timeout
TILT_DE = 2   # Direction error
TILT_OL = 1   # Current overload
TILT_TRF = 0  # Resolver fault

GEN_ENC = 7   # Encoder installed
GEN_EXEC = 6  # Executing remote initiated command
GEN_DES = 5   # Destination coords
GEN_OSLR = 4  # Override return
GEN_CWM = 3   # Moving clockwise
GEN_CCWM = 2  # Moving counter-clockwise
GEN_UPM = 1   # Moving up
GEN_DWNM = 0  # Moving down

def int_to_bytes(val):
    conv = list(val.to_bytes(2, byteorder='little', signed=True))
    return conv[0], conv[1]

def bytes_to_int(lsb, msb):
    return int.from_bytes(bytes([lsb, msb]), byteorder='little', signed=True)


class PanStatus:
    def __init__(self, status_val):
        self.cw_soft_lim = (status_val >> PAN_CWSL) & 1
        self.ccw_soft_lim = (status_val >> PAN_CCWSL) & 1
        self.cw_hard_lim = (status_val >> PAN_CWHL) & 1
        self.ccw_hard_lim = (status_val >> PAN_CCWHL) & 1
        self.timeout = (status_val >> PAN_TO) & 1
        self.direction_err = (status_val >> PAN_DE) & 1
        self.current_overload = (status_val >> PAN_OL) & 1
        self.resolver_fault = (status_val >> PAN_PRF) & 1

    def __str__(self):
        return 'PanStatus:{}'.format(vars(self))

class TiltStatus:
    def __init__(self, status_val):
        self.up_soft_lim = (status_val >> TILT_USL) & 1
        self.down_soft_lim = (status_val >> TILT_DSL) & 1
        self.up_hard_lim = (status_val >> TILT_UHL) & 1
        self.down_hard_lim = (status_val >> TILT_DHL) & 1
        self.timeout = (status_val >> TILT_TO) & 1
        self.direction_err = (status_val >> TILT_DE) & 1
        self.current_overload = (status_val >> TILT_OL) & 1
        self.resolver_fault = (status_val >> TILT_TRF) & 1

    def __str__(self):
        return 'TiltStatus:{}'.format(vars(self))


class GenStatus:
    def __init__(self, status_val):
        self.encoder_installed = (status_val >> GEN_ENC) & 1
        self.executing = (status_val >> GEN_EXEC) & 1
        self.dest_coords = (status_val >> GEN_DES) & 1
        self.override_return = (status_val >> GEN_OSLR) & 1
        self.moving_cw = (status_val >> GEN_CWM) & 1
        self.moving_ccw = (status_val >> GEN_CCWM) & 1
        self.moving_up = (status_val >> GEN_UPM) & 1
        self.moving_down = (status_val >> GEN_DWNM) & 1

    def __str__(self):
        return 'GenStatus:{}'.format(vars(self))


class BasicResponse:
    def __init__(self, data: list):
        print(data)
        self.pan_coord = bytes_to_int(data.pop(0), data.pop(0))   # PAN = -3600 to +3600 = -360.0 deg to +360.0 deg
        self.tilt_coord =  bytes_to_int(data.pop(0), data.pop(0))  # TILT = -1800 to +1800 = -180.0 deg to +180.0 deg

        self.pan_status = PanStatus(data.pop(0))
        self.tilt_status = TiltStatus(data.pop(0))
        self.gen_status = GenStatus(data.pop(0))

        self.zoom_cord = data.pop(0)
        self.focus_coord = data.pop(0)

        if data:
            self.cam_count = data.pop(0)
            self.cam_data = data  # Camera data is anything left

    def __str__(self):
        cam_data = 0 #'Cam:[count: {count}, data: {data}]'.format(count=self.cam_count, data=self.cam_data) if self.cam_data else ''
        return ('[Response]\n'
                'Pan coord:  {pan_coord} deg\n'
                'Tilt coord: {tilt_coord} deg\n'
                '{pan_status}\n'
                '{tilt_status}\n'
                '{gen_status}\n'
                'Zoom coord:  {zoom_coord}\n'
                'Focus coord: {focus_coord}\n'
                '{cam_data}').format(pan_coord=self.pan_coord,
                                      tilt_coord=self.tilt_coord,
                                      pan_status=self.pan_status,
                                      tilt_status=self.tilt_status,
                                      gen_status=self.gen_status,
                                      zoom_coord=self.zoom_cord,
                                      focus_coord=self.focus_coord,
                                      cam_data=cam_data)
        # return ('[Response]\n'
        #        'Pan coord:  {pan_coord} deg\n'
        #        'Tilt coord: {tilt_coord} deg\n').format(pan_coord=self.pan_coord,
        #                                      tilt_coord=self.tilt_coord)
        



# Calculate the LRC
def calc_checksum(cmd, data: list):
    partial = cmd
    for char in data:
        partial ^= char
    return partial

# escape a character (byte) if necessary. Returns a list
def escape_char(char) -> list:
    if char in [CTRL_STX, CTRL_ETX, CTRL_ACK, CTRL_NACK, CTRL_ESC]:  # If is ctrl char
        return [CTRL_ESC, char | 0x80] # return ESC + char (with 7th bit set, aka logical OR 0x80)
    return [char] # else return just the char (in a list for consistency)

# Removes encoded escape chars from a buffer. ignores beginning and ending ctrl chars
def remove_escapes(buffer):
    buffer_out = [buffer[0]]  # ACK or NACK
    i = 1
    # While loop is necessary since increment is not constant
    while i < len(buffer) - 1:  # range [1, 1 less than end index] (excl ETX)
        if buffer[i] == CTRL_ESC:
            buffer_out.append(buffer[i+1] & 0x7f)  # Add next byte with 7th bit cleared
            i += 2  # Jump forward 2
        else:
            buffer_out.append(buffer[i])
            i += 1  # Step forward 1
    buffer_out.append(buffer[-1])  # Append last byte (ETX)

    return buffer_out

# Build list of bytes for request packet. Returns a list
def build_req(cmd, data: list) -> list:
    buffer_out = [CTRL_STX] # no escaping for STX
    buffer_out += escape_char(IDENTITY)
    buffer_out += escape_char(cmd)

    for d in data:
        buffer_out += escape_char(d)

    lrc = calc_checksum(cmd, data)
    buffer_out += escape_char(lrc)
    buffer_out += [CTRL_ETX] # no escaping for ETX

    return buffer_out

def rcv_response(serial_port) -> list:
    char_in = 0
    while char_in not in [CTRL_ACK, CTRL_NACK]:
        char_in = list(serial_port.read())[0]  # read one char, conv bytes to list, pull 1st

    buffer = [char_in]  # Add start char as first char in buffer

    while char_in != CTRL_ETX:
        char_in = list(serial_port.read())[0]  # read one char
        buffer.append(char_in)  # add to list

    return buffer


# Sends a request packet and returns the device response
def send_request(serial_port, buffer: list, get_rsp=True) -> list:
    # Format bytes as hex strings for printing
    as_hex = [hex(x) for x in buffer]
    #print('Sending bytes {}'.format(as_hex))

    as_bytes = bytes(buffer)  # build bytes object from list of numbers (buffer)
    serial_port.write(as_bytes)

    if not get_rsp:
        return None

    rsp = rcv_response(serial_port)
    as_hex = [hex(x) for x in rsp]
    #print('Received bytes {}'.format(as_hex))

    return rsp

def get_status_jog(serial_port,
                   get_response=True,
                   ru=1, osl=1, stop=0, res=0,
                   pan_speed=0, pan_dir=0,
                   tilt_speed=0, tilt_dir=0,
                   zoom_speed=0, zoom_dir=0,
                   focus_speed=0, focus_dir=0):
    jog_cmd_byte = (ru << 3) | (osl << 2) | (stop << 1) | res
    pan = (pan_speed << 1) | pan_dir
    tilt = (tilt_speed << 1) | tilt_dir
    zoom = (zoom_speed << 1) | zoom_dir
    focus = (focus_speed << 1) | focus_dir
    
    buffer = build_req(CMD_GET_STAT_JOG, [jog_cmd_byte, pan, tilt, zoom, focus])

    response_raw = send_request(serial_port, buffer, get_rsp=get_response)
    if not get_response:
        return

    response = remove_escapes(response_raw)
    print([x for x in response_raw]) 

    return_code = response.pop(0)  # Pop ACK/NACK off front
    response.pop()  # Pop ETX from back
    rsp_identity = response.pop(0)  # Pop Identity off front
    rsp_cmd = response.pop(0) # Pull cmd byte
    rsp_lrc = response.pop() # Pull LRC from back
    rsp_data = response  # Everything else is data

    # Check LRC is valid
    lrc_matches = calc_checksum(rsp_cmd, rsp_data) == rsp_lrc
    
    print(('RCV | GET STATUS/JOG | '
            'ACK: {ack_rsp}, ID: {id}, CMD: {cmd}, '
            'LRC match: {lrc_match}, Data: {data}').format(ack_rsp='YES' if return_code == CTRL_ACK else 'NO',
                                                          id=hex(rsp_identity),
                                                          cmd=hex(rsp_cmd),
                                                          lrc_match='YES' if lrc_matches else 'NO',
                                                          data=[hex(x) for x in rsp_data]))

    formatted_resp = BasicResponse(rsp_data)
    print(rsp_data)

def init_autobaud(serial_port):
    print('Initializing Autobaud')
    # Spam port with status requests one-way 20 times to be sure
    for i in range(20):
        get_status_jog(serial_port, get_response=False)

    # Give it some time to settle
    time.sleep(1)
    
    # Flush the input so we don't have to parse all that
    serial_port.flushInput()

def mv_to_coord(serial_port,pan,tilt, get_response=True):
    pan_lsb, pan_msb = int_to_bytes(pan)
    tilt_lsb, tilt_msb = int_to_bytes(tilt)
    
    buffer = build_req(CMD_MV_ENT_COORD, [pan_lsb, pan_msb, tilt_lsb, tilt_msb])

    response_raw = send_request(serial_port, buffer, get_rsp=get_response)
    if not get_response:
        return

    response = remove_escapes(response_raw)

    return_code = response.pop(0)  # Pop ACK/NACK off front
    response.pop()  # Pop ETX from back
    rsp_identity = response.pop(0)  # Pop Identity off front
    rsp_cmd = response.pop(0) # Pull cmd byte
    rsp_lrc = response.pop() # Pull LRC from back
    rsp_data = response  # Everything else is data

    # Check LRC is valid
    lrc_matches = calc_checksum(rsp_cmd, rsp_data) == rsp_lrc

    print(('RCV | MV TO Coord | '
           'ACK: {ack_rsp}, ID: {id}, CMD: {cmd}, '
           'LRC match: {lrc_match}, Data: {data}').format(ack_rsp='YES' if return_code == CTRL_ACK else 'NO',
                                                          id=hex(rsp_identity),
                                                          cmd=hex(rsp_cmd),
                                                          lrc_match='YES' if lrc_matches else 'NO',
                                                          data=[hex(x) for x in rsp_data]))

    formatted_resp = BasicResponse(rsp_data)
    print(formatted_resp)

def mv_to_abszero(serial_port,get_response=True):


    buffer = build_req(CMD_MV_ABS, [0])

    response_raw = send_request(serial_port, buffer, get_rsp=get_response)
    if not get_response:
        return

    response = remove_escapes(response_raw)

    return_code = response.pop(0)  # Pop ACK/NACK off front
    response.pop()  # Pop ETX from back
    rsp_identity = response.pop(0)  # Pop Identity off front
    rsp_cmd = response.pop(0) # Pull cmd byte
    rsp_lrc = response.pop() # Pull LRC from back
    rsp_data = response  # Everything else is data

    # Check LRC is valid
    lrc_matches = calc_checksum(rsp_cmd, rsp_data) == rsp_lrc

    print(('RCV | MV TO ABS | '
           'ACK: {ack_rsp}, ID: {id}, CMD: {cmd}, '
           'LRC match: {lrc_match}, Data: {data}').format(ack_rsp='YES' if return_code == CTRL_ACK else 'NO',
                                                          id=hex(rsp_identity),
                                                          cmd=hex(rsp_cmd),
                                                          lrc_match='YES' if lrc_matches else 'NO',
                                                          data=[hex(x) for x in rsp_data]))

    formatted_resp = BasicResponse(rsp_data)
    print(rsp_data)

def mv_to_home(serial_port,pan,tilt, get_response=True):
    
    pan_lsb, pan_msb = int_to_bytes(pan)
    tilt_lsb, tilt_msb = int_to_bytes(tilt)

    buffer = build_req(CMD_MV_HOME, [pan_lsb, pan_msb, tilt_lsb, tilt_msb])

    response_raw = send_request(serial_port, buffer, get_rsp=get_response)
    if not get_response:
        return

    response = remove_escapes(response_raw)

    return_code = response.pop(0)  # Pop ACK/NACK off front
    response.pop()  # Pop ETX from back
    rsp_identity = response.pop(0)  # Pop Identity off front
    rsp_cmd = response.pop(0) # Pull cmd byte
    rsp_lrc = response.pop() # Pull LRC from back
    rsp_data = response  # Everything else is data

    # Check LRC is valid
    lrc_matches = calc_checksum(rsp_cmd, rsp_data) == rsp_lrc

    print(('RCV | MV TO Coord | '
           'ACK: {ack_rsp}, ID: {id}, CMD: {cmd}, '
           'LRC match: {lrc_match}, Data: {data}').format(ack_rsp='YES' if return_code == CTRL_ACK else 'NO',
                                                          id=hex(rsp_identity),
                                                          cmd=hex(rsp_cmd),
                                                          lrc_match='YES' if lrc_matches else 'NO',
                                                          data=[hex(x) for x in rsp_data]))

    formatted_resp = BasicResponse(rsp_data)
    print(formatted_resp)
