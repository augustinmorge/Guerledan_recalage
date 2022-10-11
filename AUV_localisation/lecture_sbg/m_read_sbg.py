import numpy as np
import matplotlib.pyplot as plt

import ins_Sbg
import osm_ui

if __name__ == "__main__":

    directory = "/media/michel/bckMlg7b/20180703_RecetteEkinox2/"\
        + "session_0000/2018_07_03"
    filelist = ( "log_13h00.bin",
                  "log_14h00.bin",
                  "log_15h00.bin" )
    q_draw = True

    gnss_raw_file = None
    #gnss_raw_file = "toto.sbf"
    
    # Lecture des paquets
    sbg = ins_Sbg.SbgRawData(directory, filelist, gnss_raw_file, q_draw)

    if q_draw == True:
        # Affichage des résultats
        # ------------------------

        if 0:        
            # Trajectoire
            g1 = osm_ui.plot_xy(180. / np.pi * sbg.pos_nav[:,1],
                                   180. / np.pi * sbg.pos_nav[:,0],
                                   "longitude (°)", "latitude (°)",
                                   "trajectoire Gnss","r",
                                   1)
            osm_ui.plot_xy_add(g1, 180. / np.pi * sbg.pos_gps[:,1],
                                  180. / np.pi * sbg.pos_gps[:,0], "g")
            g1.legend(("ins", "gnss"))

        if 0:
            # Attitude
            g2 = osm_ui.plot_xy(sbg.t_euler,
                                   sbg.att_euler  * 180. / np.pi,
                                   "time (s)", "attitude",
                                   "attitude", nro_fig=2)
            g2.legend(("heading", "pitch", "roll"))

        if 1:
            # heading
            g3 = osm_ui.plot_xy(sbg.t_euler,
                                   sbg.att_euler[:,0] * 180. / np.pi,
                                   "time (s)", "heading (°)",
                                   "heading","r", 3)
            osm_ui.plot_xy_add(g3, sbg.t_gps_hdt,
                                  180. / np.pi * sbg.true_heading_gps_hdt, "g")
            g3.legend(("ins", "gnss"))

        if 1:
            # pitch
            g4 = osm_ui.plot_xy(sbg.t_euler,
                                   sbg.att_euler[:,1] * 180. / np.pi,
                                   "time (s)", "pitch",
                                   "pitch","r", 4)
            osm_ui.plot_xy_add(g4, sbg.t_gps_hdt,
                                  180. / np.pi * sbg.pitch_gps_hdt, "g")
            g4.legend(("ins", "gnss"))

        if 0:
            # heading accuracy
            g5 = osm_ui.plot_xy(sbg.t_euler,
                                   sbg.att_acc_euler[:,0] * 180. / np.pi,
                                   "time (s)", "std heading",
                                   "heading accuracy","r", 5)
            osm_ui.plot_xy_add(g5, sbg.t_gps_hdt,
                                  180. / np.pi * sbg.true_heading_acc_gps_hdt,
                                  "g")
            g5.legend(("ins", "gnss"))

        if 0:
            # pitch accuracy
            g6 = osm_ui.plot_xy(sbg.t_euler,
                                   sbg.att_acc_euler[:,1] * 180. / np.pi,
                                   "time (s)", "std pitch",
                                   "pitch accuracy", "r", 6)
            osm_ui.plot_xy_add(g6, sbg.t_gps_hdt,
                                  180. / np.pi * sbg.pitch_acc_gps_hdt, "g")
            g6.legend(("ins", "gnss"))

        if 0:
            # Heave
            g7 = osm_ui.plot_xy(sbg.t_ship,
                                   sbg.pos_ship[:,2],
                                   "time (s)", "heave (m)",
                                   "heave", "r", 7)
            osm_ui.plot_xy_add(g7, sbg.t_ship_delayed,
                                  sbg.pos_ship_delayed[:, 2], "g")
            g7.legend(("ins", "ins delayed"))

        if 0:
            # Altitude
            g8 = osm_ui.plot_xy(sbg.t_nav, sbg.pos_nav[:,2],
                                   "time (s)", "altitude/pilonnement(m)",
                                   "altitude/pilonnement", "r", 8)
            osm_ui.plot_xy_add(g8, sbg.t_gps,
                                  sbg.pos_gps[:, 2], "g")
            osm_ui.plot_xy_add(g8, sbg.t_ship,
                                  sbg.pos_ship[:, 2], "b")
            osm_ui.plot_xy_add(g8, sbg.t_ship_delayed,
                                  sbg.pos_ship_delayed[:, 2], "m")
            g8.legend(("alt ins", "alt gnss", "heave ins",
                       "delayed heave ins", ))

        if 0:
            # Speed
            g9 = osm_ui.plot_xy(sbg.t_ship, sbg.velocity_ship[:,0],
                                   "time (s)", "vitesse (m/s)",
                                   "vitesse longitudinale", "r", 9)
            osm_ui.plot_xy_add(g9, sbg.t_ship_delayed,
                                  sbg.velocity_ship_delayed[:, 0], "g")
            osm_ui.plot_xy_add(g9, sbg.t_gps_vel,
                                  sbg.vel_gps_vel[:, 0], "b")
            g9.legend(("ins", "ins delayed", "gps"))

        if 0:
            g10= osm_ui.plot_xy(sbg.t_ship, sbg.velocity_ship[:,1],
                                   "time (s)", "vitesse (m/s)",
                                   "vitesse transversale", "r", 10)
            osm_ui.plot_xy_add(g10, sbg.t_ship_delayed,
                                  sbg.velocity_ship_delayed[:, 1], "g")
            osm_ui.plot_xy_add(g10, sbg.t_gps_vel,
                                  sbg.vel_gps_vel[:, 1], "b")
            g10.legend(("ins", "ins delayed", "gps"))

        if 0:
            g11 = osm_ui.plot_xy(sbg.t_ship, sbg.velocity_ship[:,2],
                                    "time (s)", "vitesse (m/s)",
                                    "vitesse verticale", "r", 11)
            osm_ui.plot_xy_add(g11, sbg.t_ship_delayed,
                                  sbg.velocity_ship_delayed[:, 2], "g")
            osm_ui.plot_xy_add(g11, sbg.t_gps_vel,
                                sbg.vel_gps_vel[:, 2], "b")
            g11.legend(("ins", "ins delayed", "gps"))

        if 0:
            # Les statuts
            g12 = osm_ui.plot_xy(sbg.t_status,
                                    sbg.st_general_main_power - 0.1,
                                    "time (s)", "status", "status general",
                                    nro_fig=12)
            osm_ui.plot_xy_add\
                (g12, sbg.t_status, sbg.st_general_imu_power - 0.05)
            osm_ui.plot_xy_add\
                (g12, sbg.t_status, sbg.st_general_gps_power + 0.)
            osm_ui.plot_xy_add\
                (g12, sbg.t_status, sbg.st_general_settings + 0.05)
            osm_ui.plot_xy_add\
                (g12, sbg.t_status, sbg.st_general_temperature + 0.1)
            g12.legend(("main power", "imu power", "gps power",
                        "settings", "temperature"))

        if 0:
            g13 = osm_ui.plot_xy(sbg.t_status,
                                    sbg.st_comm_port_a_valid - 0.05,
                                    "time (s)", "status", "comm port a",
                                    nro_fig=13)
            osm_ui.plot_xy_add\
                (g13, sbg.t_status, sbg.st_comm_port_a_rx - 0.)
            osm_ui.plot_xy_add\
                (g13, sbg.t_status, sbg.st_comm_port_a_tx + 0.05)
            g13.legend(("valid", "rx", "tx"))

        if 0:
            g14 = osm_ui.plot_xy(sbg.t_status,
                                    sbg.st_comm_port_b_valid - 0.05,
                                "time (s)", "status", "comm port b",
                                    nro_fig=14)
            osm_ui.plot_xy_add\
            (g14, sbg.t_status, sbg.st_comm_port_b_rx - 0.)
            osm_ui.plot_xy_add\
                (g14, sbg.t_status, sbg.st_comm_port_b_tx + 0.05)
            g14.legend(("valid", "rx", "tx"))

        if 0:
            g15 = osm_ui.plot_xy(sbg.t_status,
                                    sbg.st_comm_port_c_valid - 0.05,
                                    "time (s)", "status", "comm port c",
                                    nro_fig=15)
            osm_ui.plot_xy_add\
                (g15, sbg.t_status, sbg.st_comm_port_c_rx - 0.)
            osm_ui.plot_xy_add\
                (g15, sbg.t_status, sbg.st_comm_port_c_tx + 0.05)
            g15.legend(("valid", "rx", "tx"))

        if 0:
            g16 = osm_ui.plot_xy(sbg.t_status,
                                    sbg.st_comm_port_d_valid - 0.05,
                                    "time (s)", "status", "comm port d",
                                    nro_fig=16)
            osm_ui.plot_xy_add\
                (g16, sbg.t_status, sbg.st_comm_port_d_rx - 0.)
            osm_ui.plot_xy_add\
                (g16, sbg.t_status, sbg.st_comm_port_d_tx + 0.05)
            g16.legend(("valid", "rx", "tx"))

        if 0:
            g17 = osm_ui.plot_xy(sbg.t_status,
                                    sbg.st_comm_port_e_valid - 0.05,
                                    "time (s)", "status", "comm port e",
                                    nro_fig=17)
            osm_ui.plot_xy_add\
                (g17, sbg.t_status, sbg.st_comm_port_e_rx - 0.)
            osm_ui.plot_xy_add\
                (g17, sbg.t_status, sbg.st_comm_port_e_tx + 0.05)
            g17.legend(("valid", "rx", "tx"))

        if 0:
            g18 = osm_ui.plot_xy(sbg.t_status,
                                sbg.st_comm_can_rx - 0.05,
                                "time (s)", "status", "comm can",
                                nro_fig=18)
            osm_ui.plot_xy_add\
                (g18, sbg.t_status, sbg.st_comm_can_tx - 0.)
            osm_ui.plot_xy_add\
                (g18, sbg.t_status, sbg.st_comm_can_bus + 0.05)
            g18.legend(("rx", "tx", "bus"))

        if 0:
            g19 = osm_ui.plot_xy(sbg.t_status,
                                    sbg.st_aiding_gps1_pos - 0.15,
                                    "time (s)", "status", "aiding",
                                    nro_fig=19)
            osm_ui.plot_xy_add\
                (g19, sbg.t_status, sbg.st_aiding_gps1_vel - 0.1)
            osm_ui.plot_xy_add\
                (g19, sbg.t_status, sbg.st_aiding_gps1_hdt - 0.05)
            osm_ui.plot_xy_add\
                (g19, sbg.t_status, sbg.st_aiding_gps1_utc + 0.)
            osm_ui.plot_xy_add\
                (g19, sbg.t_status, sbg.st_aiding_port_mag + 0.05)
            osm_ui.plot_xy_add\
                (g19, sbg.t_status, sbg.st_aiding_port_odo + 0.1)
            osm_ui.plot_xy_add\
                (g19, sbg.t_status, sbg.st_aiding_port_dvl + 0.15)
            g19.legend(("gps1 pos", "gps1 vel", "gps1 hdt", "gps1 utc",
                        "mag", "odo", "dvl"))

        if 0:
            g20 = osm_ui.plot_xy(sbg.t_time,
                                    sbg.st_clock_stable - 0.05,
                                    "time (s)", "status", "clocks",
                                    nro_fig=20)
            osm_ui.plot_xy_add\
                (g20, sbg.t_time, sbg.st_clock_status + 0.)
            osm_ui.plot_xy_add\
                (g20, sbg.t_time, sbg.st_clock_utc_sync + 0.05)
            osm_ui.plot_xy_add\
                (g20, sbg.t_time, sbg.st_clock_utc_status + 0.1)
            g20.legend(("stable", "clock status", "utc sync", "utc status"))

        if 0:
            g21 = osm_ui.plot_xy(sbg.t_imu,
                                    sbg.st_imu_com - 0.2,
                                    "time (s)", "status", "imu",
                                    nro_fig=21)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_status - 0.15)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_accel_x - 0.1)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_accel_y - 0.05)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_accel_z + 0.)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_gyro_x + 0.05)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_gyro_y + 0.1)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_gyro_z + 0.15)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_accel_in_range + 0.2)
            osm_ui.plot_xy_add\
                (g21, sbg.t_imu, sbg.st_imu_gyro_in_range + 0.25)
            g21.legend(("com", "status", "accel_x", "accel_y", "accel_z",
                        "gyro_x", "gyro_y", "gyro_z", "accel_range",
                        "gyro_range"))

        if 0:
            g22 = osm_ui.plot_xy(sbg.t_euler,
                                    sbg.st_euler_solution_mode - 0.15,
                                    "time (s)", "status", "euler",
                                    nro_fig=22)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_attitude_valid - 0.125)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_heading_valid - 0.1)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_velocity_valid - 0.075)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_position_valid - 0.05)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_vert_ref_used - 0.025)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_mag_ref_used - 0.)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_gps1_vel_used + 0.025)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_gps1_pos_used + 0.05)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_gps1_course_used + 0.075)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_gps1_hdt_used + 0.1)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_gps2_vel_used + 0.125)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_gps2_pos_used + 0.15)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_gps2_course_used + 0.175)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_gps2_hdt_used - 0.15)
            osm_ui.plot_xy_add\
                (g22, sbg.t_euler, sbg.st_euler_odo_used - 0.15)

            g22.legend(("solution_mode", "attitude val.", "heading val.",
                        "velocity val.", "position val.",
                        "vert ref used", "mag ref used",
                        "gps1 vel used", "gps1 pos used", "gps1 course used",
                        "gps1 head used", 
                        "gps2 vel used", "gps2 pos used", "gps2 course used",
                        "gps2 head used", "odo used")) 

        if 0:
            g23 = osm_ui.plot_xy(sbg.t_quat,
                                sbg.st_quat_solution_mode - 0.15,
                                    "time (s)", "status", "quaternion",
                                    nro_fig=23)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_attitude_valid - 0.125)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_heading_valid - 0.1)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_velocity_valid - 0.075)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_position_valid - 0.05)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_vert_ref_used - 0.025)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_mag_ref_used - 0.)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_gps1_vel_used + 0.025)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_gps1_pos_used + 0.05)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_gps1_course_used + 0.075)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_gps1_hdt_used + 0.1)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_gps2_vel_used + 0.125)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_gps2_pos_used + 0.15)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_gps2_course_used + 0.175)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_gps2_hdt_used - 0.15)
            osm_ui.plot_xy_add\
                (g23, sbg.t_quat, sbg.st_quat_odo_used - 0.15)

            g23.legend(("solution_mode", "attitude val.", "heading val.",
                        "velocity val.", "position val.",
                        "vert ref used", "mag ref used",
                        "gps1 vel used", "gps1 pos used", "gps1 course used",
                        "gps1 head used", 
                        "gps2 vel used", "gps2 pos used", "gps2 course used",
                        "gps2 head used", "odo used")) 

        if 0:
            g24 = osm_ui.plot_xy(sbg.t_nav,
                                    sbg.st_nav_solution_mode - 0.15,
                                    "time (s)", "status", "nav",
                                    nro_fig=24)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_attitude_valid - 0.125)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_heading_valid - 0.1)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_velocity_valid - 0.075)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_position_valid - 0.05)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_vert_ref_used - 0.025)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_mag_ref_used - 0.)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_gps1_vel_used + 0.025)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_gps1_pos_used + 0.05)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_gps1_course_used + 0.075)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_gps1_hdt_used + 0.1)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_gps2_vel_used + 0.125)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_gps2_pos_used + 0.15)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_gps2_course_used + 0.175)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_gps2_hdt_used - 0.15)
            osm_ui.plot_xy_add\
                (g24, sbg.t_nav, sbg.st_nav_odo_used - 0.15)
            
            g24.legend(("solution_mode", "attitude val.", "heading val.",
                        "velocity val.", "position val.",
                        "vert ref used", "mag ref used",
                        "gps1 vel used", "gps1 pos used", "gps1 course used",
                        "gps1 head used", 
                        "gps2 vel used", "gps2 pos used", "gps2 course used",
                        "gps2 head used", "odo used"))

        if 0:
            g25 = osm_ui.plot_xy(sbg.t_ship,
                                    sbg.st_ship_heave_valid - 0.05,
                                "time (s)", "status", "ship",
                                    nro_fig=25)
            osm_ui.plot_xy_add\
                (g25, sbg.t_ship, sbg.st_ship_heave_vel_aided - 0.)
            osm_ui.plot_xy_add\
                (g25, sbg.t_ship, sbg.st_ship_period_available + 0.05)
            osm_ui.plot_xy_add\
                (g25, sbg.t_ship, sbg.st_ship_period_valid + 0.1)
            g25.legend(("heave valid", "heave vel aided", "period avail.",
                        "period valid"))

        if 0:
            g26 = osm_ui.plot_xy(sbg.t_ship_delayed,
                                    sbg.st_ship_delayed_heave_valid - 0.05,
                                    "time (s)", "status", "ship delayed",
                                nro_fig=26)
            osm_ui.plot_xy_add\
                (g26, sbg.t_ship_delayed,
                 sbg.st_ship_delayed_heave_vel_aided - 0.)
            osm_ui.plot_xy_add\
                (g26, sbg.t_ship_delayed,
                 sbg.st_ship_delayed_period_available + 0.05)
            osm_ui.plot_xy_add\
                (g26, sbg.t_ship_delayed,
                 sbg.st_ship_delayed_period_valid + 0.1)
            g26.legend(("heave valid", "heave vel aided", "period avail.",
                        "period valid"))

        if 0:
            g27 = osm_ui.plot_xy(sbg.t_gps_vel,
                                    sbg.st_gps_vel_status,
                                    "time (s)", "status", "gps vel",
                                    nro_fig=27)
            osm_ui.plot_xy_add\
                (g27, sbg.t_gps_vel, sbg.st_gps_vel_type + 0.05)
            g27.legend(("status", "type"))

        if 0:
            g28 = osm_ui.plot_xy(sbg.t_gps,
                                    sbg.st_gps_pos_status - 0.15,
                                    "time (s)", "status", "gps pos",
                                    nro_fig=28)
            osm_ui.plot_xy_add\
                (g28, sbg.t_gps, sbg.st_gps_pos_type - 0.1)
            osm_ui.plot_xy_add\
                (g28, sbg.t_gps, sbg.st_gps_gps_l1_used - 0.05)
            osm_ui.plot_xy_add\
                (g28, sbg.t_gps, sbg.st_gps_gps_l2_used + 0.)
            osm_ui.plot_xy_add\
                (g28, sbg.t_gps, sbg.st_gps_gps_l5_used + 0.05)
            osm_ui.plot_xy_add\
                (g28, sbg.t_gps, sbg.st_gps_glo_l1_used + 0.1)
            osm_ui.plot_xy_add\
                (g28, sbg.t_gps, sbg.st_gps_glo_l2_used + 0.15)
            g28.legend(("status", "type", "gps l1", "gps l2", "gps l5",
                        "glo l1", "glo l2"))

        if 0:
            g29 = osm_ui.plot_xy(sbg.t_gps_hdt,
                                    sbg.st_gps_hdt_sol_computed - 0.05,
                                    "time (s)", "status", "gps heading",
                                    nro_fig=29)
            osm_ui.plot_xy_add\
                (g29, sbg.t_gps_hdt, sbg.st_gps_hdt_insufficient_obs + 0.)
            osm_ui.plot_xy_add\
                (g29, sbg.t_gps_hdt, sbg.st_gps_hdt_internal_error + 0.05)
            osm_ui.plot_xy_add\
                (g29, sbg.t_gps_hdt, sbg.st_gps_hdt_height_limit + 0.1)
            g29.legend(("sol. computed", "insuff. obs.", "internal err.",
                        "height limit"))
        if 0:
            g30 = osm_ui.plot_xy(sbg.t_pressure,
                                    sbg.st_pressure_valid,
                                "time (s)", "status", "pressure",
                                    nro_fig=30)
            osm_ui.plot_xy_add\
                (g30, sbg.t_pressure, sbg.st_altitude_valid + 0.05)
            g30.legend(("pressure valid", "altitude valid"))

        if 0:
            g31 = osm_ui.plot_xy(sbg.t_mag,
                                    sbg.st_mag_mag_x - 0.2,
                                    "time (s)", "status", "magnetometer",
                                    nro_fig=31)
            osm_ui.plot_xy_add\
                (g31, sbg.t_mag, sbg.st_mag_mag_y - 0.15)
            osm_ui.plot_xy_add\
                (g31, sbg.t_mag, sbg.st_mag_mag_z - 0.1)
            osm_ui.plot_xy_add\
                (g31, sbg.t_mag, sbg.st_mag_acc_x - 0.05)
            osm_ui.plot_xy_add\
                (g31, sbg.t_mag, sbg.st_mag_acc_y + 0.)
            osm_ui.plot_xy_add\
                (g31, sbg.t_mag, sbg.st_mag_acc_y + 0.1)
            osm_ui.plot_xy_add\
                (g31, sbg.t_mag, sbg.st_mag_mag_in_range + 0.15)
            osm_ui.plot_xy_add\
                (g31, sbg.t_mag, sbg.st_mag_accel_in_range + 0.2)
            osm_ui.plot_xy_add\
                (g31, sbg.t_mag, sbg.st_mag_calibration + 0.25)
            g31.legend(("mag x", "mag y", "mag z", "acc x", "acc y",
                        "acc z", "mag in range", "accel in range",
                        "mag calibration"))
        
        plt.pause(0.1)
