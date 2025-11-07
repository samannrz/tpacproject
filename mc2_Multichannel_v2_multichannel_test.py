"""pyAFM Basic example with an MC2 system.

It will configure an MCuF system, and retrieve several A-scans.

First have a look at the imports and constants definitions just below, and
start reading from the main() function to understand how things are working.

"""

import os
import socket
import sys
import pyTPAC.Tpac as Tpac
import pyTPAC.CfgMC2 as CfgMC2
import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib.pyplot import pause
from matplotlib.widgets import Button
import matplotlib

matplotlib.use('Qt5Agg')

SERVER_IP_ADDRESS = "127.0.0.1"  # IP address of AFM_API_SOCKET server
DEVICE_TYPE = Tpac.EnumHardware.eOEMMC2  # Change to your device's type.
DEVICE_IP = "192.168.1.11"  # Replace with your device's IP address

SERVER_ADDRESS_CMD = (SERVER_IP_ADDRESS, 25000)  # Command socket
SERVER_ADDRESS_DAT = (SERVER_IP_ADDRESS, 25001)  # Data reception socket


def create_encoder(a, b) -> Tpac.EncoderSpecification:
    return Tpac.EncoderSpecification(
        type=Tpac.EnumEncoderType.eEncoderQuadrature,
        divider_times_counts_per_mm=1,
        divider=1,
        input_a=a,
        input_b=b,
        direction=Tpac.EnumOEMPAEncoderDirection.eOEMPAUpDown,
        external_reset_input=Tpac.EnumDigitalInput.eDigitalInputOff
    )


def create_pulses_shapes():
    """On each awg-capable device, up to 16 arbitrary pulse shape can be stored,
       and subsequently used. This function creates list of 6 AWG pulses: one
       of each kind.This function creates list of 6 AWG pulses, each of them
       being one of the six available types.
    """
    shapes = []
    shapes.append(Tpac.AwgPulseShapeSpecification(
        shape_index=Tpac.PulseShapeIndexT.pulse_shape_0,
        shape_type=Tpac.PulseShapeT.pulse_shape_bipolar,
        precision=200.0,
        duration=0.0,
        num_periods=1,
        f0=1.0,
        df=0.0,
        bwr=0.0,
        duty_cycle=50.0,
        num_points=0,
        points_list=[]))

    shapes.append(Tpac.AwgPulseShapeSpecification(
        shape_index=Tpac.PulseShapeIndexT.pulse_shape_1,
        shape_type=Tpac.PulseShapeT.pulse_shape_burst,
        precision=200.0,
        duration=0.0,
        num_periods=5,
        f0=5.0,
        df=0.0,
        bwr=0.0,
        duty_cycle=50.0,
        num_points=0,
        points_list=[]))
    shapes.append(Tpac.AwgPulseShapeSpecification(
        shape_index=Tpac.PulseShapeIndexT.pulse_shape_2,
        shape_type=Tpac.PulseShapeT.pulse_shape_chirp,
        precision=100.0,
        duration=5.0,
        num_periods=0.0,
        f0=5.0,
        df=1.0,
        bwr=0.0,
        duty_cycle=0.0,
        num_points=0.0,
        points_list=[]))
    shapes.append(Tpac.AwgPulseShapeSpecification(
        shape_index=Tpac.PulseShapeIndexT.pulse_shape_3,
        shape_type=Tpac.PulseShapeT.pulse_shape_gaussian,
        precision=100.0,
        duration=0.0,
        num_periods=0.0,
        f0=5.0,
        df=0.0,
        bwr=50.0,
        duty_cycle=0.0,
        num_points=0,
        points_list=[]))
    shapes.append(Tpac.AwgPulseShapeSpecification(
        shape_index=Tpac.PulseShapeIndexT.pulse_shape_4,
        shape_type=Tpac.PulseShapeT.pulse_shape_unipolar,
        precision=200.0,
        duration=0.0,
        num_periods=0,
        f0=5.0,
        df=0.0,
        bwr=0.0,
        duty_cycle=0.0,
        num_points=0,
        points_list=[]))
    points = []
    for _ in range(5):
        points.append(-1.0)
    for _ in range(5):
        points.append(1.0)
    for _ in range(1000):
        points.append(0.0)
    shapes.append(Tpac.AwgPulseShapeSpecification(
        shape_index=Tpac.PulseShapeIndexT.pulse_shape_5,
        shape_type=Tpac.PulseShapeT.pulse_shape_custom,
        precision=100.0,
        duration=0.0,
        num_periods=0.0,
        f0=0.0,
        df=0.0,
        bwr=0.0,
        duty_cycle=0.0,
        num_points=len(points),
        points_list=points))
    return shapes


def create_device_setting() -> Tpac.DeviceSettingsSpecificationMC2:
    """Create the settings structure attached to one device."""
    shapes = create_pulses_shapes()
    return Tpac.DeviceSettingsSpecificationMC2(
        trigger_mode=Tpac.EnumOEMPATrigger.eOEMPAInternal,
        trigger_step_mm=1.0,
        bit_size=Tpac.EnumBitSizeMC2.e16bits,
        scan_encoder=create_encoder(Tpac.EnumDigitalInput.eDigitalInput01,
                                    Tpac.EnumDigitalInput.eDigitalInput02),
        index_encoder=create_encoder(Tpac.EnumDigitalInput.eDigitalInput03,
                                     Tpac.EnumDigitalInput.eDigitalInput04),
        debouncer_time_us=1.0,
        negative_pulse_voltage=-200,
        bipolar_pulse_positive_voltage=50,
        bipolar_pulse_negative_voltage=50,
        probe_center_frequency_mhz=1.0,
        num_digital_outputs=0,
        digital_outputs=[],
        external_sequence_signal=Tpac.EnumDigitalInput.eDigitalInputOff,
        external_cycle_signal=Tpac.EnumDigitalInput.eDigitalInputOff,
        ascan_enable=1,
        ascan_request=Tpac.EnumAscanRequest.eAscanAll,
        ascan_request_freq=100.0,
        attenuator_20db=0,
        analog_gain_db=30.0,
        request_io=Tpac.EnumOEMPARequestIO.eOEMPAOnCycleOnly,
        awg_mode_enabled=1,
        num_pulses_shapes=len(shapes),
        pulses_shapes=shapes,
        external_reset_all_encoders=Tpac.EnumDigitalInput.eDigitalInputOff
    )


def create_transmission(enabled) -> Tpac.ChannelTransmissionSpecificationMC2:
    print('creating transmission')

    return Tpac.ChannelTransmissionSpecificationMC2(
        enabled=1,
        wedge_delay_us=0.0,
        pulse_delay_us=0.0,
        pulse_width_us=0.5,
        period=5,
        pulse_count=1,
        pulse_index=Tpac.PulseShapeIndexT.pulse_shape_0

    )


def create_dac() -> Tpac.DacSpecificationMC2:
    return Tpac.DacSpecificationMC2(
        enable=0,
        num_points=0,
        tofs=[],
        gains=[],
        auto_stop=0,
        tracking_enabled=0,
        tracking_gate_index=0,
        tracking_gate_cycle=0,
        tracking_gate_channel=0
    )


def create_gate(i: int) -> Tpac.GateSpecificationMC2:
    start_us = 12.63 if i == 1 else 12.69 if i == 2 else 0.0 #12.63
    stop_us = 12.91 if i == 1 else 12.77 if i == 2 else 5.0
    threshold_pct = 25.0 if i == 1 else 60.0 if i == 2 else 50.0
    if i == 1:
        rectification = Tpac.EnumRectification.eUnsignedPositive
    else:
        rectification = Tpac.EnumRectification.eSigned
    return Tpac.GateSpecificationMC2(
        enabled=0,
        start_us=start_us,
        stop_us=stop_us,
        threshold_pct=threshold_pct,
        rectification=rectification,
        amp_mode=Tpac.EnumGateModeAmp.eAmpAbsolute,
        tof_mode=Tpac.EnumGateModeTof.eTofAmplitudeDetection,
        tracking_start_enable=0,
        tracking_stop_enable=0,
        tracking_start_gate=0,
        tracking_start_cycle=0,
        tracking_start_channel=0,
        tracking_stop_gate=0,
        tracking_stop_cycle=0,
        tracking_stop_channel=0
    )


def create_reception(enabled) -> Tpac.ChannelReceptionSpecificationMC2:
    print('creating reception')
    return Tpac.ChannelReceptionSpecificationMC2(
        enabled=1,
        wedge_delay_us=0.0,
        gain_digital_db=00.0,
        gain_normalization_db=0.0,
        start_us=0.0,
        range_us=100.0,
        point_factor=5,
        point_count=0,
        compression=Tpac.EnumCompressionType.eDecimation,
        rectification=Tpac.EnumRectification.eSigned,
        #filter_index=Tpac.EnumOEMPAFilterIndex.eOEMPAFilterOff,
        filter_index=Tpac.EnumOEMPAFilterIndex(10)
,

        dac=create_dac(),
        num_gates=4,
        # Up to 4 gates can be defined
        gates=[create_gate(1),create_gate(2),create_gate(3),create_gate(4)],
        ascan_tracking_enabled=0,
        ascan_tracking_gate_index=0,
        ascan_tracking_channel_index=0,
        ascan_tracking_cycle_index=0)


def create_acquisition(emission_num, reception_num) -> CfgMC2.AcqMC2:
    # Create an array of size num_emission - 1 with disable // same for reception
    transmissions = [create_transmission(1) if i in emission_num else create_transmission(0)
                     for i in range(max(emission_num) + 1)]
    receptions = [create_reception(1) if i in reception_num else create_reception(0)
                  for i in range(max(reception_num) + 1)]
    return CfgMC2.AcqMC2(transmissions,
                         receptions,
                         time_slot_us=850.0,
                         awg_bipolar_pulsers_selected=1)


def create_filter() -> CfgMC2.FilterDefinition:
    return CfgMC2.FilterDefinition(Tpac.DigitalFilterBand.band_pass,
                                   attenuation_db=40.0,
                                   freq_a_MHz=0.58,
                                   freq_b_MHz=1.22)


def decode_data_frame(s: socket.socket) -> int:
    """
    Retrieve one data frame and decode it.

    :return: 0 in the general case, and 1 if it’s an A-scan
    """

    # pyTPAC provides a decode_frame() function, which returns frames (of type
    # Tpac.F) one by one
    frm = Tpac.decode_frame(s)

    # The kind of frame can be retrieved from the frame header:
    if frm.header.client_protocol_type == Tpac.F.ProtocolType.heartbeat:
        # This kind of void frame is received at a low pace.
        # In this example, we just ignore it.
        # print("Heartbeat", flush=True)
        return 0
    else:
        b = frm.body
        # Another way to check the frame type, is to check the class of the
        # returned frame body.
        if isinstance(b, Tpac.F.AosErrorCallbackInfoFrame):
            # Note that the current function is generic: it could be used with
            # the data socket or with the events and messages socket. In our
            # case, we’ll never receive any AosErrorCallbackInfoFrame, because
            # we are connected with the data frame. This branch won’t be
            # reached.
            print("Error frame:", end='', flush=True)
            if bool(b.has_error):
                print(f" {b.error_code}: {b.error_msg}", flush=True)
            else:
                print(" No error", flush=True)
                return 0
        elif isinstance(b, Tpac.F.AosHardwareStatusInfoFrame):
            # Note that the current function is generic: it could be used with
            # the data socket or with the events and messages socket. In our
            # case, we’ll never receive any AosHardwareStatusInfoFrame, because
            # we are connected with the data frame. This branch won’t be
            # reached.
            print(f"Status: position={b.position}, temp={b.temp}, "
                  f"lost ascans={b.lost_ascans}, lost cscans = {b.lost_cscans}, "
                  f"lost packets={b.lost_packets}", flush=True)
            return 0
        elif isinstance(b, Tpac.F.AosAscanFrame):
            # Here, we are demonstrating how we can deconstruct the
            # AosAscanFrame.
            print("Ascan: ", {'acq_id': b.acq_id,
                              'sequence_id': b.sequence_id,
                              'length_sequence_data': b.length_sequence_data,
                              'num_cycles': b.num_cycles,
                              'max_temperature': b.max_temperature,
                              'lost_ascan_count': b.lost_ascan_count,
                              'lost_cscan_count': b.lost_cscan_count,
                              'lost_packet_count': b.lost_packet_count,
                              'encoder1_count': b.encoder1_count,
                              'encoder2_count': b.encoder2_count,
                              'cycle_offset': b.cycle_offset,
                              'ios': [io.content for io in b.digital_inputs],
                              'timestamp': b.device_timestamp,
                              'nb channels': b.nb_active_channels,
                              'ascan length': b.ascan_length_active_channel}, flush=True)

            # In the case of multichannel acquisitions, the data are provided
            # as a b.length_sequence_data bytes buffer: b.sequence_data.
            #
            # It contains the active reception channels A-scans.
            #
            # The actual decoding of the A-scan is demonstrated and explained
            # in the on_refresh_ascan() callback of testGui-MC2.py program
            return 1
        else:
            # Should not be reached
            print(f"Unknown: {frm.header.client_protocol_type.name}, "
                  f"{frm.header.body_size} bytes", flush=True)
            return 0


def store_data_frame(s: socket.socket,max_flush=1) -> int:
    """
    Retrieve one data frame and decode it.

    :return: 0 in the general case, and 1 if it’s an A-scan
    """

    # pyTPAC provides a decode_frame() function, which returns frames (of type
    # Tpac.F) one by one
    for _ in range(max_flush):
        try:
            frm = Tpac.decode_frame(s)
        except Exception:
            # No more data or socket closed
            break
    #frm = Tpac.decode_frame(s)
    nested_data = []
    nested_metadata=[]
    # The kind of frame can be retrieved from the frame header:
    if frm.header.client_protocol_type == Tpac.F.ProtocolType.heartbeat:
        # This kind of void frame is received at a low pace.
        # In this example, we just ignore it.
        # print("Heartbeat", flush=True)
        return (0, 0, 0, 0)
    else:
        b = frm.body
        # Another way to check the frame type, is to check the class of the
        # returned frame body.
        if isinstance(b, Tpac.F.AosErrorCallbackInfoFrame):
            # Note that the current function is generic: it could be used with
            # the data socket or with the events and messages socket. In our
            # case, we’ll never receive any AosErrorCallbackInfoFrame, because
            # we are connected with the data frame. This branch won’t be
            # reached.
            print("Error frame:", end='', flush=True)
            if bool(b.has_error):
                print(f" {b.error_code}: {b.error_msg}", flush=True)
            else:
                print(" No error", flush=True)
                return (0, 0, 0, 0)
        elif isinstance(b, Tpac.F.AosHardwareStatusInfoFrame):
            # Note that the current function is generic: it could be used with
            # the data socket or with the events and messages socket. In our
            # case, we’ll never receive any AosHardwareStatusInfoFrame, because
            # we are connected with the data frame. This branch won’t be
            # reached.
            print(f"Status: position={b.position}, temp={b.temp}, "
                  f"lost ascans={b.lost_ascans}, lost cscans = {b.lost_cscans}, "
                  f"lost packets={b.lost_packets}", flush=True)
            return (0, 0, 0, 0)
        elif isinstance(b, Tpac.F.AosAscanFrame):
            # Here, we are demonstrating how we can deconstruct the
            # AosAscanFrame.
            # Amplitude conversion and normalization
            # factor is : (2^(N) - 1) / 200, where N is the number of significant bits
            num_activated_channels = len(b.active_channel_indexes)
            conversion = {
                1: {'typ': 'i1', 'factor': 1.275, 'offset': 0.5},
                2: {'typ': 'i2', 'factor': 327.675, 'offset': 0.5},
                4: {'typ': 'i4', 'factor': 671088.635, 'offset': 0.5}
            }
            for channel in range(num_activated_channels):
                metadata = {'acq_id': b.acq_id, 'sequence_id': b.sequence_id,
                            'num_cycles': b.num_cycles, 'lost_ascan_count': b.lost_ascan_count,
                            'lost_packet_count': b.lost_packet_count, 'cycle_offset': b.cycle_offset,
                            'nb channels': b.nb_active_channels, 'ascan length': b.ascan_length_active_channel[channel],
                            'active_channel_index': b.active_channel_indexes[channel]}

                offset = sum(b.ascan_length_active_channel[0:channel]) * b.channels_data_type_size[channel]
                end_read = offset + b.ascan_length_active_channel[channel] * b.channels_data_type_size[channel]

                # TODO check with several rectifications
                cv = conversion[b.channels_data_type_size[channel]]
                data = np.frombuffer(b.raw_sequence_data[offset:end_read], dtype=cv['typ'])

                nested_data.append(data)
                nested_metadata.append(metadata)

            # In the case of multichannel acquisitions, the data are provided
            # as a b.length_sequence_data bytes buffer: b.sequence_data.
            #
            # It contains the active reception channels A-scans.
            #
            # The actual decoding of the A-scan is demonstrated and explained
            # in the on_refresh_ascan() callback of testGui-MC2.py program
            return nested_data, nested_metadata, 1, cv
        else:
            # Should not be reached
            print(f"Unknown: {frm.header.client_protocol_type.name}, "
                  f"{frm.header.body_size} bytes", flush=True)

            return 0


stop_acquisition = False


def stop(event):
    global stop_acquisition
    stop_acquisition = True
    print("Acquisition stopped. Disconnecting from pulser...")


axes_dict = {}
figures_dict = {}

def emitted_pulse(f0, num_periods, fs ):
    # fs in Mhz
    #total time in µs
    # Convert precision to total number of samples
    T = 1 / f0  # period in µs
    total_time = num_periods * T

    # Time vector in seconds
    t = np.linspace(0, total_time, int(fs * total_time), endpoint=False)

    # Generate bipolar square wave
    pulse = np.sign(np.sin(2 * np.pi * f0 * t))
    return pulse

def Manolakis(xe,ye,ze):
    from numpy.linalg import inv, det

    xe = np.array(xe, dtype=float)
    ye = np.array(ye, dtype=float)
    ze = np.array(ze, dtype=float)

    # differences between emitters
    x21, y21, z21 = xe[1] - xe[0], ye[1] - ye[0], ze[1] - ze[0]
    x31, y31, z31 = xe[2] - xe[0], ye[2] - ye[0], ze[2] - ze[0]

    # squared distances from origin
    S = np.sqrt([xe[0] ** 2 + ye[0] ** 2 + ze[0] ** 2,
                 xe[1] ** 2 + ye[1] ** 2 + ze[1] ** 2,
                 xe[2] ** 2 + ye[2] ** 2 + ze[2] ** 2])

    delta = np.array([S[0] ** 2 - S[1] ** 2, S[0] ** 2 - S[2] ** 2]).reshape(2, 1)
    # vertical differences vector d (2x1)
    d = np.array([[z21], [z31]]).reshape(2, 1)  # shape (2,1)

    # horizontal (x,y) of station 1
    rh1 = np.array([[xe[0]], [ye[0]]]).reshape(2, 1)  # (2,1)

    W = np.array([[x21, y21],
                  [x31, y31]])  # shape (2,2)
    G = inv(W.T) @ inv(W)  # shape (2,2)

    a = 1.0 + (d.T @ G @ d)  # shape (1,1) (matrix), convert to scalar
    a = a.item()

    # --- lam (4×1) coefficients ---
    # compute intermediate terms carefully
    lam0 = -(2 * rh1.T @ inv(W) @ d - 2 * ze[0] + d.T @ (inv(W)).T @ inv(W) @ delta) / (2 * a)
    lam0 = lam0.item()
    lam1 = (z21 * (G[0, 0] + G[0, 1]) + z31 * (G[1, 1] + G[0, 1])) / (2 * a)
    lam2 = -(z21 * G[0, 0] + z31 * G[0, 1]) / (2 * a)
    lam3 = -(z31 * G[1, 1] + z21 * G[0, 1]) / (2 * a)

    lam = np.array([lam0, lam1, lam2, lam3], dtype=float).reshape(4, 1)
    lambda3 = lam

    invW = inv(W)
    w1 = invW[0, :].reshape(1, 2)  # shape (2,)
    w2 = invW[1, :].reshape(1, 2)  # shape (2,)

    detW = det(W)

    lambda1_term2 = np.array(
        [[(-w1 @ delta / 2).item()], [((y31 - y21) / (2 * detW)).item()], [(-y31 / (2 * detW)).item()],
         [(y21 / (2 * detW)).item()]])
    lambda1_term1 = (-w1 @ d).item() * lam
    lambda1 = lambda1_term1 + lambda1_term2

    lambda2_term2 = np.array(
        [[(-w2 @ delta / 2).item()], [((x21 - x31) / (2 * detW)).item()], [(x31 / (2 * detW)).item()],
         [(-x21 / (2 * detW)).item()]])
    lambda2_term1 = (-w2 @ d).item() * lam
    lambda2 = lambda2_term1 + lambda2_term2
    # Construct a "lambda matrix" (3 x 4) like MATLAB's lambda
    # Rows: lambda1', lambda2', lambda3' (each is 4-element row)
    Lambda_mat = np.vstack([lambda1.reshape(1, 4), lambda2.reshape(1, 4), lambda3.reshape(1, 4)])  # shape (3,4)

    L = Lambda_mat[:, 1:].copy()
    # Diagonal matrices L1,L2,L3 formed from elements 2..4 of lambda1,2,3
    L1 = np.diag(lambda1[1:4])
    L2 = np.diag(lambda2[1:4])
    L3 = np.diag(lambda3[1:4])

    # --- ksi coefficients (10 values) ---
    # compute the scalar inner products carefully (convert 1x1 arrays to float)
    invW_term = invW.T @ invW

    ksi0 = lam0 ** 2 - ((delta.T @ invW_term @ delta) / 4.0 + (rh1.T @ invW @ delta).item() + S[0] ** 2) / a
    ksi0 = ksi0.item()
    e = delta.T @ inv(W.T) @ invW / 2 + rh1.T @ invW
    # e is shape (1,2) earlier - convert elements
    e1 = float(e[0, 0])
    e2 = float(e[0, 1])

    ksi1 = 2.0 * lam0 * lam1 + (1.0 + e1 + e2) / a
    ksi2 = 2.0 * lam0 * lam2 - e1 / a
    ksi3 = 2.0 * lam0 * lam3 - e2 / a
    ksi4 = lam1 ** 2 - (G[0, 0] + 2.0 * G[0, 1] + G[1, 1]) / (4.0 * a)
    ksi5 = lam2 ** 2 - G[0, 0] / (4.0 * a)
    ksi6 = lam3 ** 2 - G[1, 1] / (4.0 * a)
    ksi7 = 2.0 * lam1 * lam2 + (G[0, 0] + G[0, 1]) / (2.0 * a)
    ksi8 = 2.0 * lam1 * lam3 + (G[1, 1] + G[0, 1]) / (2.0 * a)
    ksi9 = 2.0 * lam2 * lam3 - G[0, 1] / (2.0 * a)

    ksi = np.array([ksi0, ksi1, ksi2, ksi3, ksi4, ksi5, ksi6, ksi7, ksi8, ksi9], dtype=float).T

    # --- mu vectors ---
    mu_plus = np.array([- (w1 @ d).item(), - (w2 @ d).item(), 1]).reshape(3, 1)
    mu_minus = -mu_plus

    return Lambda_mat, ksi, mu_plus

from scipy.optimize import least_squares
def trilaterate_nls(emitters, distances, x0=None):
    """
    emitters: (N,3) array of emitter coordinates [(x,y,z)...], N>=3
    distances: (N,) array of distances from unknown receiver to each emitter
    x0: optional initial guess (3,)
    returns: estimated position (3,)
    """
    emitters = np.asarray(emitters, dtype=float)
    distances = np.asarray(distances, dtype=float)
    assert emitters.shape[0] == distances.size

    if x0 is None:
        # sensible initial guess: weighted mean of emitters
        x0 = emitters.mean(axis=0)
        x0[2]= 62

    def residuals(p):
        d_model = np.linalg.norm(emitters - p, axis=1)
        return d_model - distances

    res = least_squares(residuals, x0, method='lm')  # 'lm' or 'trf'
    return res.x, res

import numpy as np
from scipy.optimize import least_squares

def trilaterate_shared_x(emitters, distances_all, x0=None):
    """
    Trilaterate positions of multiple receivers that share the same x coordinate.

    emitters: (N, 3) array of emitter coordinates [(x, y, z)...]
    distances_all: (M, N) array of measured distances [receiver_i, emitter_j]
    x0: optional initial guess for optimization

    Returns:
        receivers: (M, 3) array of estimated receiver positions
        res: least_squares result object
    """
    emitters = np.asarray(emitters, dtype=float)
    distances_all = np.asarray(distances_all, dtype=float)
    M, N = distances_all.shape

    # Initial guess
    if x0 is None:
        mean_x = emitters[:, 0].mean()
        mean_y = emitters[:, 1].mean()
        mean_z = emitters[:, 2].mean()
        x0 = np.array([mean_x] + [mean_y, mean_z] * M)

    def residuals(vars):
        x_shared = vars[0]
        res_list = []
        for i in range(M):
            yi, zi = vars[1 + 2*i:1 + 2*i + 2]
            Ri = np.array([x_shared, yi, zi])
            d_model = np.linalg.norm(emitters - Ri, axis=1)
            res_list.append(d_model - distances_all[i])
        return np.concatenate(res_list)

    res = least_squares(residuals, x0, method='lm')
    vars_opt = res.x
    x_shared = vars_opt[0]
    receivers = np.array([[x_shared, vars_opt[1 + 2*i], vars_opt[1 + 2*i + 1]] for i in range(M)])
    return receivers, res



def main():
    """Main function."""

    #### Server connection
    ######################

    print(f"Trying to connect to the server [IP,PORT:{SERVER_ADDRESS_CMD}]")
    try:
        aas = Tpac.Tpac(SERVER_ADDRESS_CMD, timeout=None)

        # Better to clean AFM API state after connection.
        aas.shut_down()

    except OSError as e:
        print(f"x Failed to connect to the command server: is it running?"
              f" ({SERVER_ADDRESS_CMD}): ", e)
        sys.exit(0)

    except Exception:
        print(f"x Failed to connect to the command server ({SERVER_ADDRESS_CMD}).")
        sys.exit(0)

    print(f"- Connected with AFM_API_SOCKET server {SERVER_ADDRESS_CMD}.")

    #### Device connection
    ######################

    print(f"Creating a {DEVICE_TYPE.name} device.")
    # With pyTPAC, it’s your responsability to provide the handles (ids) which
    # will refer to AFM API objects. Here, we are creating one 'Hardware',
    # which will be referenced as 'hw_id=1'.
    hw_id = 1
    ans = aas.create_hw(hw_id=hw_id, hw_type=DEVICE_TYPE)
    # Almost all pyTPAC functions return a dict.
    print(ans)

    print(f"Connecting the device to the IP {DEVICE_IP}.")
    ans = aas.connect(hw_id=hw_id, address=DEVICE_IP)
    print(ans)

    # Here, we are looking for the 'enable' element of the returned dictionnary
    # in order to check if the connection was successfull.
    isConnected = aas.is_connected(hw_id)['enable']

    if isConnected:
        print(f"- hardware successfully connected ({DEVICE_IP})")
    else:
        print("x Connection error")
        aas.close()
        return

    awg_available = aas.is_awg_available(hw_id)['enable'] == 1
    print(f"Is awg available on his hardware? {awg_available}")

    ##### Configuration loading
    filters_definition=[create_filter()] # another create filter for other cycles
    for idx, f in enumerate(filters_definition):
     aas.set_filter_definition(hw_id=hw_id,
                               filter_index=Tpac.EnumOEMPAFilterIndex(10),
                               filter_band=Tpac.DigitalFilterBand(f.filter_band),
                               attenuation_dB=f.attenuation_db,
                               freq_a_MHz=f.freq_a_MHz,
                               freq_b_MHz=f.freq_b_MHz)

    # register the acquisition_callback (this last step is instructing AFM API
    # that we want to receive A-scan dataframes for those acquisitions)
    acquisitions = [create_acquisition([0], [0,1,2]),
                    create_acquisition([1], [0,1,2]),
                    create_acquisition([2], [0,1,2])]

    for acq_id, acq in enumerate(acquisitions):
        mca = Tpac.MultichannelAcquisition(num_receive_channels=len(acq.receptions),
                                           receptions=acq.receptions,
                                           num_transmit_channels=len(acq.transmissions),
                                           transmissions=acq.transmissions,
                                           time_slot_us=acq.time_slot_us,
                                           awg_bipolar_pulsers_selected=acq.awg_bipolar_pulsers_selected,
                                           analog_gain=20.0)

        acqui = Tpac.AcqSpec(acq_type=Tpac.AcquisitionType.multichannel,
                             num_cycles=3,
                             acq_id=acq_id,
                             multichannel_acquisition_spec=mca)

        ans = aas.register_acquisition(acq_id=acq_id,
                                       hw_id=hw_id,
                                       acquisition=acqui)

        aas.register_callback_acquisition(acq_id=acq_id)

    # This is the number of acquisitions in cfg.
    nb_acqs = len(acquisitions)

    # We decided to use every acquisitions from the cfg file. Here we generate
    # a list with the acquisitions ids. We could program and use only a subset
    # of them. Another solution would have been to write the above loop as a
    # list comprehension returning the acquisition ids list.
    acquisitions_list = list(range(nb_acqs))
    device_setting = create_device_setting()
    # The configure_hw_MC2 call is really programming the hardware with the
    # acquisitions we registered before
    aas.configure_hw_MC2(hw_id=hw_id,
                         device_settings=device_setting,
                         write_file=1)
    # aas.configure_hw_MC2(hw_id,
    #                       device_setting,
    #                       num_acquisitions=nb_acqs,
    #                       acquisition_ids=acquisitions_list,
    #                       write_file = False)

    #### Opening a socket: will be used to gather data
    ##################################################

    try:
        sdata = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sdata.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sdata.connect(SERVER_ADDRESS_DAT)
        print(f"- Connected to the data server ({SERVER_ADDRESS_DAT} OK)")
    except OSError as e:
        print(f"x Data server connection failure ({SERVER_ADDRESS_DAT}): ",
              end='')
        print(e)
        aas.disconnect(hw_id)
        aas.close()
        return 0

    #### Start acquisition
    ######################

    isStarted = aas.enable_pulser(hw_id)['hw_id'] == hw_id
    if isStarted:
        print("- enable_pulser success")
    else:
        print("x enable_pulser failure")
        aas.disconnect(hw_id)
        aas.close()
        return

    #### Compute the time at wich the acquisition will be stopped
    #############################################################

    compteur_ascan = 0

    #### Receiving data with decode_data_frame() function
    #####################################################
    ds_factor = 1
    fs = 100 #Mhz
    point_factor = 5
    range_us = 100 #us
    fft_update_every = 20
    T = 20
    c0 = round(1402.40 + 5.01 * T - 0.055 * T ** 2 + 0.00022 * T ** 3, 1)
    fs_r = fs / point_factor

    plt.ion()  # Turn on interactive mode
    fft_fig = None  # placeholder for the FFT figure

    # Create a 3x3 figure (rows = emitters, columns = receivers)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))  # 3 emitters x 3 receivers

    # Each subplot can have up to 3 lines (for multichannel acquisitions)
    lines = [[None for _ in range(3)] for _ in range(3 * 3)]  # 9 subplots × 3 channels
    D_ToF = np.full((3, 3), 0,dtype=float)
    pos3D_mano = np.full((3, 3), None)
    pos3D_nls = np.full((3, 3), None)
    peak_markers = [[None for _ in range(3)] for _ in range(3)]
    zero_markers = [[None for _ in range(3)] for _ in range(3)]
    # Set up each subplot
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.set_xlim(0,range_us)

            ax.set_ylim(-30, 30)
            ax.grid(True)
            ax.set_title(f"Emitter {row} | Receiver {col}")

    fig.suptitle("Real-Time Data Plots")
    plt.subplots_adjust(bottom=0.2, hspace=0.4)

    # Add a stop button
    stop_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
    stop_button = Button(stop_ax, 'Stop')
    stop_button.on_clicked(stop)

    # ===== Main acquisition loop =====
    max_itnum = 3
    itnum = 0
    while not stop_acquisition: #and itnum<max_itnum:
        itnum+=1
        nested_data, nested_metadata, x, cv = store_data_frame(sdata, 160)
        # print((nested_metadata[1]))
        samples = np.arange(len(nested_data[0]))/fs_r
        samples = samples[::ds_factor]


        if x >= 0:  # Only process A-scan frames
            for ch_idx, metadata in enumerate(nested_metadata):
                emitter = metadata['acq_id']  # row
                receiver = metadata['active_channel_index']  # column
                # print(emitter,receiver)
                ax = axes[emitter, receiver]

                ## plot the received Signals
                # Create line if it does not exist
                line_idx = emitter * 3 + receiver
                if lines[line_idx][ch_idx] is None:
                    lines[line_idx][ch_idx], = ax.plot([], [], lw=1)

                downsampled_data = nested_data[ch_idx][::ds_factor]
                lines[line_idx][ch_idx].set_data(samples, downsampled_data / cv['factor'])

                ## end of signal plots

                from scipy.signal import find_peaks
                ##
                # plt.figure()
                # pulse = emitted_pulse(1,1,fs_r)
                # plt.plot(np.arange(len(pulse))/fs_r,pulse)
                signal = nested_data[ch_idx]/cv['factor']
                signal = np.copy(signal)
                signal_norm = signal / np.max(signal)
                signal_norm[0:int(10 * fs_r)] = 0
                peak,_ = find_peaks(signal_norm, prominence=1e-1)
                peak = peak[0]
                zero_idx = peak - 0.25 * fs_r

                D_ToF[emitter, receiver] = (zero_idx/(fs_r*1e6)) * c0 * 1e3
                print(f"Emitter: {emitter}, Receiver: {receiver}, D_ToF: {D_ToF[emitter, receiver]:.3f} mm")

                if peak_markers[emitter][receiver] is None:
                    peak_markers[emitter][receiver], = ax.plot(
                        peak / fs_r, signal[peak], 'bo', markersize=6
                    )
                else:
                    peak_markers[emitter][receiver].set_data([peak/ fs_r], [signal[peak]])

                # Update or create the red zero-cross marker
                if zero_markers[emitter][receiver] is None:
                    zero_markers[emitter][receiver], = ax.plot(
                        zero_idx / fs_r, signal[int(zero_idx)], 'ro', markersize=6
                    )
                else:
                    zero_markers[emitter][receiver].set_data([zero_idx / fs_r], [signal[int(zero_idx)]])







            fig.canvas.draw_idle()
            fig.canvas.flush_events()  # Update GUI
            plt.gcf().canvas.flush_events()

            plt.pause(0.0001)
        ##### plotting fft
            # if itnum % fft_update_every == 0:
            #     fft_sig = np.abs(np.fft.rfft(nested_data[0]))
            #     freq_ax = np.fft.rfftfreq(len(nested_data[0]), 1 / (point_factor))
            #
            #     # Create FFT figure only once
            #     if fft_fig is None:
            #         fft_fig, fft_ax = plt.subplots()
            #         fft_ax.set_xlabel("Frequency (MHz)")
            #         fft_ax.set_ylabel("Magnitude")
            #         fft_ax.grid(True)
            #         fft_ax.set_title("FFT of channel 0 (update every 5 frames)")
            #
            #     fft_ax.clear()  # clear previous FFT
            #     fft_ax.plot(freq_ax, fft_sig)  # convert Hz → MHz
            #     fft_fig.canvas.draw_idle()
            #     fft_fig.canvas.flush_events()
            for ir in range(3):
                R1, R2, R3 = D_ToF[0, ir], D_ToF[1, ir], D_ToF[2, ir]
                u = np.array([1, R1 ** 2, R2 ** 2, R3 ** 2]).reshape(4, 1)
                v = np.array(
                    [1, R1 ** 2, R2 ** 2, R3 ** 2, R1 ** 4, R2 ** 4, R3 ** 4, R1 ** 2 * R2 ** 2, R1 ** 2 * R3 ** 2,
                     R2 ** 2 * R3 ** 2]).reshape(10, 1)

                e1 = [4 * np.sqrt(3), 4, 0]
                e2 = [0, -8, 0]
                e3 = [-4 * np.sqrt(3), 4, 0]

                Lambda_mat, ksi, mu_plus = Manolakis([e1[0],e2[0],e3[0]], [e1[1],e2[1],e3[1]], [e1[2],e2[2],e3[2]])
                # print(Lambda_mat)
                # print(mu_plus)
                # print(ksi)
                # computed target position
                rc_plus = (Lambda_mat @ u) + (mu_plus * np.sqrt((ksi.T @ v).item()))
                pos3D_mano[ir, :] = np.ravel(rc_plus)
                print('Position using Monalokis: ', pos3D_mano)

                # pos3D_nls[ir], res_nls = trilaterate_nls([e1, e2, e3], [R1, R2, R3])

                # print(R1, R2, R3)
                # print('Position using NLS: ', pos3D_nls[ir], ' res error: ', res_nls.fun)

                # receivers_est, res = trilaterate_shared_x([e1, e2, e3], [R1, R2, R3])
                #
                #
                # print("Estimated receiver positions:\n", receivers_est)
                # print("Residual RMS error:", np.sqrt(np.mean(res.fun ** 2)))
            ### To activate if you want to stock data and save it
            #  data_list.append(data)  # Append the vector to the list
            #  metadata_list.append(metadata)

    plt.ioff()
    plt.close(fig)



    #### Stop pulser
    ################

    ans = aas.disable_pulser(hw_id)
    print(ans)
    isStopped = True  # aas.disable_pulser(hw_id)['hw_id'] == hw_id

    if isStopped:
        print("- disable_pulser success")
    else:
        print("x disable_pulser failure")
        aas.disconnect(hw_id)
        aas.close()
        return

    sdata.close()

    #### Disconnect
    ###############

    if aas.disconnect(hw_id)['hw_id'] == hw_id:
        print("- Hardware successfully disconnected")
    else:
        print("x Failure during hardware disconnection")
        aas.close()
        return

    aas.close()


if __name__ == '__main__':
    main()
