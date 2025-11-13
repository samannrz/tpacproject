
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

def Manolakis_Souha(xe,ye,ze):
    # Relative vectors
    x21, x31 = xe[1] - xe[0], xe[2] - xe[0]
    y21, y31 = ye[1] - ye[0], ye[2] - ye[0]
    z21, z31 = ze[1] - ze[0], ze[2] - ze[0]

    # Geometry matrix
    W = np.array([[x21, y21], [x31, y31]])
    invW = np.linalg.inv(W)
    detW = np.linalg.det(W)
    invWT = invW.T
    G = invWT @ invW

    S = np.sum(np.stack([xe, ye, ze])**2, axis=0)**0.5
    delta = np.array([S[0]**2 - S[1]**2, S[0]**2 - S[2]**2])
    d = np.array([z21, z31])
    rh1 = np.array([xe[0], ye[0]])

    delta_half = delta / 2
    e = (delta_half @ invWT @ invW) + (rh1 @ invW)
    a = 1 + (d @ G @ d)

    lam0 = -(2 * (rh1 @ invW @ d) - 2 * ze[0] + d @ G @ delta) / (2 * a)
    lam1 = (z21 * (G[0, 0] + G[0, 1]) + z31 * (G[1, 1] + G[0, 1])) / (2 * a)
    lam2 = -(z21 * G[0, 0] + z31 * G[0, 1]) / (2 * a)
    lam3 = -(z31 * G[1, 1] + z21 * G[0, 1]) / (2 * a)

    lam = np.array([lam0, lam1, lam2, lam3])
    lambda_all = np.array([
        (-invW[0] @ d) * lam + np.array([-invW[0] @ delta_half,
                                        (y31 - y21) / (2 * detW),
                                        -y31 / (2 * detW),
                                        y21 / (2 * detW)]),
        (-invW[1] @ d) * lam + np.array([-invW[1] @ delta_half,
                                        (x21 - x31) / (2 * detW),
                                        x31 / (2 * detW),
                                        -x21 / (2 * detW)]),
        lam
    ])

    # ksi terms
    ksi = np.zeros(10)
    ksi[0] = lam0**2 - (delta @ G @ delta_half + rh1 @ invW @ delta + S[0]**2) / a
    ksi[1] = 2 * lam0 * lam1 + (1 + e[0] + e[1]) / a
    ksi[2] = 2 * lam0 * lam2 - e[0] / a
    ksi[3] = 2 * lam0 * lam3 - e[1] / a
    ksi[4] = lam1**2 - (G[0, 0] + 2 * G[0, 1] + G[1, 1]) / (4 * a)
    ksi[5] = lam2**2 - G[0, 0] / (4 * a)
    ksi[6] = lam3**2 - G[1, 1] / (4 * a)
    ksi[7] = 2 * lam1 * lam2 + (G[0, 0] + G[0, 1]) / (2 * a)
    ksi[8] = 2 * lam1 * lam3 + (G[1, 1] + G[0, 1]) / (2 * a)
    ksi[9] = 2 * lam2 * lam3 - G[0, 1] / (2 * a)

    ksi = ksi[:, np.newaxis]
    mu_plus = np.array([-invW[0] @ d, -invW[1] @ d, 1])[:, np.newaxis]

    return lambda_all, ksi, mu_plus


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



def main_temp():
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

def main():

    R1, R2, R3 = 45,45,45
    #R1, R2, R3 = 45.2,46.2,45.2
    #R1, R2, R3 = 45.4,46.4,45.4

    u = np.array([1, R1 ** 2, R2 ** 2, R3 ** 2]).reshape(4, 1)
    v = np.array(
        [1, R1 ** 2, R2 ** 2, R3 ** 2, R1 ** 4, R2 ** 4, R3 ** 4, R1 ** 2 * R2 ** 2, R1 ** 2 * R3 ** 2,
         R2 ** 2 * R3 ** 2]).reshape(10, 1)

    e1 = [4 * np.sqrt(3), 4, 0]
    e2 = [0, -8, 0]
    e3 = [-4 * np.sqrt(3), 4, 0]

    #Lambda_mat, ksi, mu_plus = Manolakis_Souha([e1[0], e2[0], e3[0]], [e1[1], e2[1], e3[1]], [e1[2], e2[2], e3[2]])
    Lambda_mat, ksi, mu_plus = Manolakis([e1[0], e2[0], e3[0]], [e1[1], e2[1], e3[1]], [e1[2], e2[2], e3[2]])
    print('Saman: ', Lambda_mat, ksi, mu_plus)
    print(Lambda_mat)
    print(mu_plus)
    print(ksi)
    #computed target position
    rc_plus = (Lambda_mat @ u) + (mu_plus * np.sqrt((ksi.T @ v).item()))
    pos3D_mano = np.ravel(rc_plus)
    print('Position using Monalokis: ', pos3D_mano)


if __name__ == '__main__':
    main()
