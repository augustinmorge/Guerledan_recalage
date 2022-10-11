#!/usr/bin/env python3
import os
import sys
import struct
import datetime

import numpy as np

import osm_ui
import osm_utils

class SbgImu:
    """ Version de l'Ekinox2 (firmware plus récent que celui de l'Ellipse)"""

    def __init__(self, directory, filelist=None,
                 check_synchro=None, ref_time=None):

        # Si filelist est vide : on cherche tous les fichiers dispos
        # dans le repértoire (avec le suffixe .bin et dans l'ordre
        # temporel
        if filelist is None:
            filelist = []
            for x in os.scandir(directory):
                if x.is_dir():
                    for xx in os.scandir(x.path):
                        if xx.is_file() and xx.name.endswith(".bin"):
                            filelist.append(os.path.join(x.name, xx.name))
                elif x.is_file() and x.name.endswith(".bin"):
                    filelist.append(x.name)

            filelist = sorted(filelist)

        self.ref_time = ref_time
        self.directory = directory
        self.filelist = filelist

        self.check_synchro = check_synchro

        self.type_set = set()

        if type( filelist ) not in (tuple, list):
            filelist = (filelist,)

        self.f_list = []
        for filename in filelist:
            self.f = open(os.path.join(directory, filename), 'rb')
            self.size = os.path.getsize(os.path.join(directory, filename))

            self.f_list.append(self.f)

        self.offset = []
        if self.check_synchro is not None:
            n = 10000
            for f in self.f_list:
                str_ = f.read(n)
                offset = str_.find(check_synchro)
                if offset == -1:
                    osm_ui.print_("synchro is not found")
                osm_ui.print_(offset)
                f.seek(offset, os.SEEK_SET)
            self.offset.append( offset )

        self.f_number = 0
        self.f = self.f_list[0]

    def rewind( self ):

        for i_, f in enumerate(self.f_list):
            self.f = f
            if len(self.offset) == 0:
                self.f.seek(0, os.SEEK_SET)
            else:
                self.f.seek(self.offset[i_], os.SEEK_SET)

        self.f_number = 0
        self.f = self.f_list[0]

    def close(self):
        for i_, f in enumerate( self.f_list ):
            if f is not None:
                f.close()
            self.f_list[ i_ ] = None
        self.f_number = 0

    def read_packet(self, wished_packet=None):
        while 1:
            try:
                sync1, sync2, msg, class_, len_ \
                    = struct.unpack("<4BH", self.f.read(6))

                # Resynchronisation
                if sync1 != 255 or sync2 != 90:
                    osm_ui.print_("pb synchro")
                    n = 10000
                    str_ = self.f.read(n)
                    offset = str_.find(b"\xffZ")
                    if offset == -1:
                        osm_ui.print_("synchro is not found")
                        raise EOFError
                    else:
                        osm_ui.print_("Resynchronisation {:d}".format(offset))
                        self.f.seek(-n+offset, os.SEEK_CUR)
                        sync1, sync2, msg, class_, len_ \
                            = struct.unpack("<4BH", self.f.read(6))

            except struct.error:
                osm_ui.print_(self.filelist[self.f_number])
                self.f_number += 1
                if self.f_number == len( self.f_list ):
                    raise EOFError
                self.f = self.f_list[self.f_number]
                try:
                    sync1, sync2, msg, class_, len_ \
                        = struct.unpack("<4BH", self.f.read(6))
                except struct.error:
                    osm_ui.print_("Second try to read a file is unsuccessful")
                    raise EOFError

            assert class_ == 0
            if msg not in self.type_set:
                self.type_set.add(msg)

            if wished_packet is None or msg in wished_packet:

                if msg == 0:
                    # Fast internal IMU
                    self.fast_imu_data = {}
                    self.fast_imu_data["time stamp"],\
                        self.fast_imu_data["status"],\
                        self.fast_imu_data["accel x"],\
                        self.fast_imu_data["accel y"],\
                        self.fast_imu_data["accel z"],\
                        self.fast_imu_data["gyro x"],\
                        self.fast_imu_data["gyro y"],\
                        self.fast_imu_data["gyro z"],\
                        = struct.unpack("<IH6h", self.f.read(18))

                elif msg == 1:
                    self.status_ = {}
                    if len_ == 22:
                        self.status_["time stamp"],\
                            self.status_["general status"],\
                            _,\
                            self.status_["com status"],\
                            self.status_["aiding status"],\
                            _,\
                            _,\
                            = struct.unpack("<I2H3IH", self.f.read(22))
                    else:
                        self.status_["time stamp"],\
                            self.status_["general status"],\
                            _,\
                            self.status_["com status"],\
                            self.status_["aiding status"],\
                            _,\
                            _,\
                            self.status_["up time"],\
                            = struct.unpack("<I2H3IHI", self.f.read(26))
                elif msg == 2:
                    self.utc_time = {}
                    self.utc_time["time stamp"],\
                        self.utc_time["clock status"],\
                        self.utc_time["year"],\
                        self.utc_time["month"],\
                        self.utc_time["day"],\
                        self.utc_time["hour"],\
                        self.utc_time["min"],\
                        self.utc_time["sec"],\
                        self.utc_time["nanosec"],\
                        self.utc_time["gps tow"],\
                        = struct.unpack("<I2H5B2I", self.f.read(21))

                elif msg == 3:
                    self.imu_data = {}
                    self.imu_data["time stamp"],\
                    self.imu_data["imu status"],\
                    self.imu_data["accel x"],\
                    self.imu_data["accel y"],\
                    self.imu_data["accel z"],\
                    self.imu_data["gyro x"],\
                    self.imu_data["gyro y"],\
                    self.imu_data["gyro z"],\
                    self.imu_data["temp"],\
                    self.imu_data["delta vx"],\
                    self.imu_data["delta vy"],\
                    self.imu_data["delta vz"],\
                    self.imu_data["delta angle x"],\
                    self.imu_data["delta angle y"],\
                    self.imu_data["delta angle z"],\
                    = struct.unpack("<IH13f", self.f.read(58))

                elif msg == 4:
                    self.mag_ = {}
                    self.mag_["time stamp"],\
                        self.mag_["mag status"],\
                        self.mag_["mag x"],\
                        self.mag_["mag y"],\
                        self.mag_["mag z"],\
                        self.mag_["accel x"],\
                        self.mag_["accel y"],\
                        self.mag_["accel z"],\
                        = struct.unpack("<IH6f", self.f.read(30))

                elif msg == 5:
                    self.mag_calib_ = {}
                    self.mag_calib_["time stamp"],\
                        _,\
                    self.mag_calib_["buffer"],\
                    = struct.unpack("<IH16s", self.f.read(22))

                elif msg == 6:
                    self.ekf_euler = {}
                    self.ekf_euler["time stamp"], \
                        self.ekf_euler["roll"], \
                        self.ekf_euler["pitch"], \
                        self.ekf_euler["yaw"], \
                        self.ekf_euler["roll acc"], \
                        self.ekf_euler["pitch acc"], \
                        self.ekf_euler["yaw acc"], \
                        self.ekf_euler["solution status"], \
                        =struct.unpack("<I6fI", self.f.read(32))

                elif msg == 7:
                    self.ekf_quat = {}
                    self.ekf_quat["time stamp"], \
                        self.ekf_quat["w"], \
                        self.ekf_quat["x"], \
                        self.ekf_quat["y"], \
                        self.ekf_quat["z"], \
                        self.ekf_quat["roll acc"], \
                        self.ekf_quat["pitch acc"], \
                        self.ekf_quat["yaw acc"], \
                        self.ekf_quat["solution status"], \
                         =struct.unpack("<I7fI", self.f.read(36))

                elif msg == 8:
                    self.ekf_nav = {}
                    self.ekf_nav["time stamp"], \
                        self.ekf_nav["velocity n"], \
                        self.ekf_nav["velocity e"], \
                        self.ekf_nav["velocity d"], \
                        self.ekf_nav["velocity n acc"], \
                        self.ekf_nav["velocity e acc"], \
                        self.ekf_nav["velocity d acc"], \
                        self.ekf_nav["latitude"], \
                        self.ekf_nav["longitude"], \
                        self.ekf_nav["altitude"], \
                        self.ekf_nav["undulation"], \
                        self.ekf_nav["latitude acc"], \
                        self.ekf_nav["longitude acc"], \
                        self.ekf_nav["altitude acc"], \
                        self.ekf_nav["solution status"], \
                        = struct.unpack("<I6f3d4fI", self.f.read(72))

                    self.ekf_nav["longitude"] *= np.pi / 180.
                    self.ekf_nav["latitude"] *= np.pi / 180.

                elif msg == 9:
                    self.ship_motion = {}
                    self.ship_motion["time stamp"],\
                        self.ship_motion["heave period"],\
                        self.ship_motion["surge"],\
                        self.ship_motion["sway"],\
                        self.ship_motion["heave"],\
                        self.ship_motion["accel x"],\
                        self.ship_motion["accel y"],\
                        self.ship_motion["accel z"],\
                        self.ship_motion["vel x"],\
                        self.ship_motion["vel y"],\
                        self.ship_motion["vel z"],\
                        self.ship_motion["status"],\
                        = struct.unpack("<I10fH", self.f.read(46))

                elif msg == 13:
                    self.gps1_vel = {}
                    self.gps1_vel["time stamp"],\
                        self.gps1_vel["gps vel status"],\
                        self.gps1_vel["gps tow"],\
                        self.gps1_vel["vel n"],\
                        self.gps1_vel["vel e"],\
                        self.gps1_vel["vel d"],\
                        self.gps1_vel["vel acc n"],\
                        self.gps1_vel["vel acc e"],\
                        self.gps1_vel["vel acc d"],\
                        self.gps1_vel["course"],\
                        self.gps1_vel["course acc"],\
                        = struct.unpack("<3I8f", self.f.read(44))

                elif msg == 14:
                    self.gps1_pos = {}
                    self.gps1_pos["time stamp"],\
                        self.gps1_pos["gps pos status"],\
                        self.gps1_pos["gps tow"],\
                        self.gps1_pos["latitude"],\
                        self.gps1_pos["longitude"],\
                        self.gps1_pos["altitude"],\
                        self.gps1_pos["undulation"],\
                        self.gps1_pos["pos acc lat"],\
                        self.gps1_pos["pos acc lon"],\
                        self.gps1_pos["pos acc alt"],\
                        self.gps1_pos["num sv used"],\
                        self.gps1_pos["base station id"],\
                        self.gps1_pos["diff age"],\
                        = struct.unpack("<3I3d4fB2H", self.f.read(57))

                    self.gps1_pos["latitude"] *= np.pi / 180.
                    self.gps1_pos["longitude"] *= np.pi / 180.


                elif msg == 15:
                    self.gps1_hdt = {}
                    self.gps1_hdt["time stamp"],\
                        self.gps1_hdt["hdt status"],\
                        self.gps1_hdt["gps tow"],\
                        self.gps1_hdt["gps true heading"],\
                        self.gps1_hdt["gps true heading acc"],\
                        self.gps1_hdt["gps pitch"],\
                        self.gps1_hdt["gps pitch acc"],\
                        = struct.unpack("<IHI4f", self.f.read(26))


                elif msg == 16:
                    self.gps2_vel = {}
                    self.gps2_vel["time stamp"],\
                        self.gps2_vel["gps vel status"],\
                        self.gps2_vel["gps tow"],\
                        self.gps2_vel["vel n"],\
                        self.gps2_vel["vel e"],\
                        self.gps2_vel["vel d"],\
                        self.gps2_vel["vel acc n"],\
                        self.gps2_vel["vel acc e"],\
                        self.gps2_vel["vel acc d"],\
                        self.gps2_vel["course"],\
                        self.gps2_vel["course acc"],\
                        = struct.unpack("<3I8f", self.f.read(44))

                elif msg == 17:
                    self.gps2_pos = {}
                    self.gps2_pos["time stamp"],\
                        self.gps2_pos["gps pos status"],\
                        self.gps2_pos["gps tow"],\
                        self.gps2_pos["latitude"],\
                        self.gps2_pos["longitude"],\
                        self.gps2_pos["altitude"],\
                        self.gps2_pos["undulation"],\
                        self.gps2_pos["pos acc lat"],\
                        self.gps2_pos["pos acc lon"],\
                        self.gps2_pos["pos acc alt"],\
                        self.gps2_pos["num sv used"],\
                        self.gps2_pos["base station id"],\
                        self.gps2_pos["diff age"],\
                        = struct.unpack("<3I3d4fB2H", self.f.read(57))

                    self.gps1_pos["latitude"] *= np.pi / 180.
                    self.gps1_pos["longitude"] *= np.pi / 180.


                elif msg == 18:
                    self.gps2_hdt = {}
                    self.gps2_hdt["time stamp"],\
                        self.gps2_hdt["hdt status"],\
                        self.gps2_hdt["gps tow"],\
                        self.gps2_hdt["gps true heading"],\
                        self.gps2_hdt["gps true heading acc"],\
                        self.gps2_hdt["gps pitch"],\
                        self.gps2_hdt["gps pitch acc"],\
                        = struct.unpack("<IHI4f", self.f.read(26))


                elif msg == 19:
                    self.odo_vel = {}
                    self.odo_vel["time stamp"],\
                        self.odo_vel["odo status"],\
                        self.odo_vel["odo vel"],\
                        =struct.unpack("<IHf", self.f.read(10))

                elif msg == 24:
                    self.event_A = {}
                    self.event_A["time stamp"],\
                        self.event_A["event status"],\
                        self.event_A["time offset 0"],\
                        self.event_A["time offset 1"],\
                        self.event_A["time offset 2"],\
                        self.event_A["time offset 3"],\
                        = struct.unpack("<I5H", self.f.read(14))

                elif msg == 25:
                    self.event_B = {}
                    self.event_B["time stamp"],\
                        self.event_B["event status"],\
                        self.event_B["time offset 0"],\
                        self.event_B["time offset 1"],\
                        self.event_B["time offset 2"],\
                        self.event_B["time offset 3"],\
                        = struct.unpack("<I5H", self.f.read(14))

                elif msg == 26:
                    self.event_C = {}
                    self.event_C["time stamp"],\
                        self.event_C["event status"],\
                        self.event_C["time offset 0"],\
                        self.event_["time offset 1"],\
                        self.event_C["time offset 2"],\
                        self.event_C["time offset 3"],\
                        = struct.unpack("<I5H", self.f.read(14))

                elif msg == 27:
                    self.event_D = {}
                    self.event_D["time stamp"],\
                        self.event_D["event status"],\
                        self.event_D["time offset 0"],\
                        self.event_D["time offset 1"],\
                        self.event_D["time offset 2"],\
                        self.event_D["time offset 3"],\
                        = struct.unpack("<I5H", self.f.read(14))

                elif msg == 31:
                    self.gps1_raw  = self.f.read(len_)


                elif msg == 32:
                    self.ship_delayed_motion = {}
                    self.ship_delayed_motion["time stamp"],\
                        self.ship_delayed_motion["heave period"],\
                        self.ship_delayed_motion["surge"],\
                        self.ship_delayed_motion["sway"],\
                        self.ship_delayed_motion["heave"],\
                        self.ship_delayed_motion["accel x"],\
                        self.ship_delayed_motion["accel y"],\
                        self.ship_delayed_motion["accel z"],\
                        self.ship_delayed_motion["vel x"],\
                        self.ship_delayed_motion["vel y"],\
                        self.ship_delayed_motion["vel z"],\
                        self.ship_delayed_motion["status"],\
                        = struct.unpack("<I10fH", self.f.read(46))

                elif msg == 36:
                    self.pressure = {}
                    self.pressure["time stamp"],\
                        self.pressure["altimeter status"],\
                        self.pressure["pressure"],\
                        self.pressure["altitude"],\
                        = struct.unpack("<IH2f", self.f.read(14))

                elif msg == 38:
                    self.gps2_raw  = self.f.read(len_)

                elif msg == 44:
                    self.imu_short_data = {}
                    self.imu_short_data["time stamp"],\
                    self.imu_short_data["imu status"],\
                    self.imu_short_data["delta vx"],\
                    self.imu_short_data["delta vy"],\
                    self.imu_short_data["delta vz"],\
                    self.imu_short_data["delta angle x"],\
                    self.imu_short_data["delta angle y"],\
                    self.imu_short_data["delta angle z"],\
                    self.imu_short_data["temperature"],\
                    = struct.unpack("<IH6ih", self.f.read(32))

                    # Conversion des données en flottant
                    self.imu_short_data["delta vx"] /= 1048576.
                    self.imu_short_data["delta vy"] /= 1048576.
                    self.imu_short_data["delta vz"] /= 1048576.
                    self.imu_short_data["delta angle x"] /= 67108864.
                    self.imu_short_data["delta angle y"] /= 67108864.
                    self.imu_short_data["delta angle z"] /= 67108864.
                    self.imu_short_data["temperature"] /= 256.


                else:
                    osm_ui.print_("unknown message {:d}".format(msg))
                    self.f.seek(len_, os.SEEK_CUR)

                crc, ext = struct.unpack("<HB", self.f.read(3))
                #print("ext {:x}".format( ext ))
                return msg
            else:
                self.f.seek(len_+3, os.SEEK_CUR)

class SbgRawData:
    def __init__(self, directory, filelist, gnss_raw_file=None,
                 q_load_data=False, ref_time=None):
        self.ref_time = ref_time

        n_status = 0
        n_time = 0
        n_imu = 0
        n_euler = 0
        n_quat = 0
        n_pressure = 0
        n_nav = 0
        n_ship = 0
        n_ship_delayed = 0
        n_mag = 0
        n_gps_vel = 0
        n_gps = 0
        n_gps_hdt = 0
        n_gps_raw = 0

        sbg = SbgImu(directory, filelist, None, ref_time)

        if gnss_raw_file is not None:
            f_gnss = open(gnss_raw_file, "wb")
        else:
            f_gnss = None

        while 1:
            try:
                type_ = sbg.read_packet()

                if type_ == 1:
                    n_status += 1
                elif type_ == 2:
                    if n_time == 0 and self.ref_time is None:
                        self.ref_time = datetime.datetime\
                                   (year = sbg.utc_time["year"],
                                    month = sbg.utc_time["month"],
                                    day = sbg.utc_time["day"],
                                    hour = sbg.utc_time["hour"],
                                    minute = sbg.utc_time["min"],
                                    second = sbg.utc_time["sec"])
                    n_time += 1
                elif type_ == 3:
                    n_imu += 1
                elif type_ == 4:
                    n_mag += 1
                elif type_ == 6:
                    n_euler += 1
                elif type_ == 7:
                    n_quat += 1
                elif type_ == 8:
                    n_nav += 1
                elif type_ == 9:
                    n_ship += 1
                elif type_ == 13:
                    n_gps_vel += 1
                elif type_ == 14:
                    n_gps += 1
                elif type_ == 15:
                    n_gps_hdt += 1
                elif type_ == 31:
                    n_gps_raw += 1
                    if f_gnss is not None:
                        f_gnss.write(sbg.gps1_raw)
                elif type_ == 32:
                    n_ship_delayed += 1
                elif type_ == 36:
                    n_pressure += 1


            except (EOFError, struct.error):
                break

        osm_ui.print_("type messages enregistrés :")
        for x in sbg.type_set:
            osm_ui.print_(x)

        if f_gnss is not None:
            f_gnss.close()

        if q_load_data == True:
            sbg.rewind()

            self.t_status = np.empty((n_status,), np.float64)
            self.status = np.empty((n_status,3), np.uint32)

            self.t_time = np.empty((n_time,2), np.float64)
            self.status_time = np.empty((n_time,), np.uint32)
            self.tow_gps_time = np.empty((n_time,), np.float32)

            self.t_imu = np.empty((n_imu,), np.float64)
            self.status_imu = np.empty((n_imu,), np.uint32)
            self.accel_imu = np.empty((n_imu,3), np.float32)
            self.gyro_imu = np.empty((n_imu,3), np.float32)
            self.temp_imu = np.empty((n_imu,), np.float32)
            self.delta_v_imu = np.empty((n_imu,3), np.float32)
            self.delta_angle_imu = np.empty((n_imu,3), np.float32)

            self.t_euler = np.empty((n_euler,), np.float64)
            self.att_euler = np.empty((n_euler,3), np.float32)
            self.att_acc_euler = np.empty((n_euler,3), np.float32)
            self.status_euler = np.empty((n_euler,), np.uint32)

            self.t_quat = np.empty((n_quat,), np.float64)
            self.att_quat = np.empty((n_quat,4), np.float32)
            self.att_acc_quat = np.empty((n_quat,3), np.float32)
            self.status_quat = np.empty((n_quat,), np.uint32)

            self.t_nav = np.empty((n_nav,), np.float64)
            self.velocity_nav = np.empty((n_nav,3), np.float32)
            self.velocity_acc_nav = np.empty((n_nav,3), np.float32)
            self.pos_nav = np.empty((n_nav,3), np.float64)
            self.pos_acc_nav = np.empty((n_nav,3), np.float32)
            self.geoid_nav = np.empty((n_nav,), np.float32)
            self.status_nav = np.empty((n_nav,), np.uint32)

            self.t_ship = np.empty((n_ship,), np.float64)
            self.heave_period_ship  = np.empty((n_ship,), np.float32)
            self.pos_ship = np.empty((n_ship,3), np.float32)
            self.accel_ship = np.empty((n_ship,3), np.float32)
            self.velocity_ship = np.empty((n_ship,3), np.float32)
            self.status_ship = np.empty((n_ship,), np.uint32)

            self.t_ship_delayed = np.empty((n_ship_delayed,), np.float64)
            self.heave_period_ship_delayed\
                = np.empty((n_ship_delayed,), np.float32)
            self.pos_ship_delayed = np.empty((n_ship_delayed,3), np.float32)
            self.accel_ship_delayed = np.empty((n_ship_delayed,3), np.float32)
            self.velocity_ship_delayed = np.empty((n_ship_delayed,3), np.float32)
            self.status_ship_delayed = np.empty((n_ship_delayed,), np.uint32)

            self.t_mag = np.empty((n_mag,), np.float64)
            self.mag  = np.empty((n_mag, 3), np.float32)
            self.accel_mag = np.empty((n_mag,3), np.float32)
            self.status_mag = np.empty((n_mag,), np.uint32)

            self.t_gps_vel = np.empty((n_gps_vel,), np.float64)
            self.vel_gps_vel  = np.empty((n_gps_vel,3), np.float32)
            self.vel_acc_gps_vel = np.empty((n_gps_vel,3), np.float32)
            self.tow_gps_vel =np.empty((n_gps_vel,), np.float32)
            self.course_gps_vel =np.empty((n_gps_vel,), np.float32)
            self.course_acc_gps_vel =np.empty((n_gps_vel,), np.float32)
            self.status_gps_vel = np.empty((n_gps_vel,), np.uint32)

            self.t_gps = np.empty((n_gps,), np.float64)
            self.tow_gps =np.empty((n_gps,), np.float32)
            self.pos_gps = np.empty((n_gps,3), np.float64)
            self.pos_acc_gps = np.empty((n_gps,3), np.float32)
            self.geoid_gps = np.empty((n_gps,), np.float32)
            self.status_gps = np.empty((n_gps,), np.uint32)
            self.num_sv_gps = np.empty((n_gps,), np.uint8)
            self.base_station_gps = np.empty((n_gps,), np.uint16)
            self.diff_age_gps = np.empty((n_gps,), np.float32)

            self.t_gps_hdt = np.empty((n_gps_hdt,), np.float64)
            self.tow_gps_hdt =np.empty((n_gps_hdt,), np.float32)
            self.status_gps_hdt = np.empty((n_gps_hdt,), np.uint32)
            self.true_heading_gps_hdt = np.empty((n_gps_hdt,), np.float32)
            self.true_heading_acc_gps_hdt = np.empty((n_gps_hdt,), np.float32)
            self.pitch_gps_hdt = np.empty((n_gps_hdt,), np.float32)
            self.pitch_acc_gps_hdt = np.empty((n_gps_hdt,), np.float32)

            self.t_pressure = np.empty((n_pressure,), np.float64)
            self.status_pressure = np.empty((n_pressure,), np.uint32)
            self.pressure = np.empty((n_pressure,), np.float32)
            self.altitude_pressure = np.empty((n_pressure,), np.float32)

            n_status = 0
            n_time = 0
            n_imu = 0
            n_euler = 0
            n_quat = 0
            n_nav = 0
            n_ship = 0
            n_ship_delayed = 0
            n_mag = 0
            n_gps_vel = 0
            n_gps = 0
            n_gps_hdt = 0
            n_pressure = 0

            while 1:
                try:
                    type_ = sbg.read_packet()

                    if type_ == 1:
                        self.t_status[n_status] = sbg.status_["time stamp"]
                        self.status[n_status, 0] = sbg.status_["general status"]
                        self.status[n_status, 1] = sbg.status_["com status"]
                        self.status[n_status, 2] = sbg.status_["aiding status"]
                        n_status += 1


                    elif type_ == 2:

                        self.t_time[n_time, 0] = sbg.utc_time["time stamp"]
                        t = (datetime.datetime\
                             (year = sbg.utc_time["year"],
                              month = sbg.utc_time["month"],
                              day = sbg.utc_time["day"],
                              hour = sbg.utc_time["hour"],
                              minute = sbg.utc_time["min"],
                              second = sbg.utc_time["sec"],
                              microsecond\
                              = sbg.utc_time["nanosec"]//1000) - self.ref_time)\
                              .total_seconds()
                        self.t_time[n_time, 1] = t
                        self.status_time[n_time] = sbg.utc_time["clock status"]
                        self.tow_gps_time[n_time]\
                            = sbg.utc_time["gps tow"] / 1000.
                        n_time += 1

                    elif type_ == 3:

                        self.t_imu[n_imu] = sbg.imu_data["time stamp"]
                        self.status_imu[n_imu] = sbg.imu_data["imu status"]
                        self.accel_imu[n_imu, 0] = sbg.imu_data["accel x"]
                        self.accel_imu[n_imu, 1] = sbg.imu_data["accel y"]
                        self.accel_imu[n_imu, 2] = sbg.imu_data["accel z"]
                        self.gyro_imu[n_imu, 0] = sbg.imu_data["gyro x"]
                        self.gyro_imu[n_imu, 1] = sbg.imu_data["gyro y"]
                        self.gyro_imu[n_imu, 2] = sbg.imu_data["gyro z"]
                        self.temp_imu[n_imu] = sbg.imu_data["temp"]
                        self.delta_v_imu[n_imu, 0] = sbg.imu_data["delta vx"]
                        self.delta_v_imu[n_imu, 1] = sbg.imu_data["delta vy"]
                        self.delta_v_imu[n_imu, 2] = sbg.imu_data["delta vz"]
                        self.delta_angle_imu[n_imu, 0] \
                            = sbg.imu_data["delta angle x"]
                        self.delta_angle_imu[n_imu, 1] \
                            = sbg.imu_data["delta angle y"]
                        self.delta_angle_imu[n_imu, 2] \
                            = sbg.imu_data["delta angle z"]
                        n_imu += 1

                    elif type_ == 4:

                        self.t_mag[n_mag] = sbg.mag["time stamp"]
                        self.mag[n_mag, 0] = sbg.mag["mag x"]
                        self.mag[n_mag, 1] = sbg.mag["mag y"]
                        self.mag[n_mag, 2] = sbg.mag["mag z"]
                        self.accel_mag[n_mag, 0] = sbg.mag["accel x"]
                        self.accel_mag[n_mag, 1] = sbg.mag["accel y"]
                        self.accel_mag[n_mag, 2] = sbg.mag["accel z"]
                        self.status_mag[n_mag] = sbg.mag["mag status"]
                        n_mag += 1

                    elif type_ == 6:

                        self.t_euler[n_euler] = sbg.ekf_euler["time stamp"]
                        self.att_euler[n_euler, 2] = sbg.ekf_euler["roll"]
                        self.att_euler[n_euler, 1] = sbg.ekf_euler["pitch"]
                        self.att_euler[n_euler, 0] = sbg.ekf_euler["yaw"]
                        self.att_acc_euler[n_euler, 2]\
                            = sbg.ekf_euler["roll acc"]
                        self.att_acc_euler[n_euler, 1]\
                            = sbg.ekf_euler["pitch acc"]
                        self.att_acc_euler[n_euler, 0]\
                            = sbg.ekf_euler["yaw acc"]
                        self.status_euler[n_euler] \
                            = sbg.ekf_euler["solution status"]
                        n_euler += 1


                    elif type_ == 7:

                        self.t_quat[n_quat] = sbg.ekf_quat["time stamp"]
                        self.att_quat[n_quat, 0] = sbg.ekf_quat["w"]
                        self.att_quat[n_quat, 1] = sbg.ekf_quat["x"]
                        self.att_quat[n_quat, 2] = sbg.ekf_quat["y"]
                        self.att_quat[n_quat, 3] = sbg.ekf_quat["z"]
                        self.att_acc_quat[n_quat, 2] = sbg.ekf_quat["roll acc"]
                        self.att_acc_quat[n_quat, 1] = sbg.ekf_quat["pitch acc"]
                        self.att_acc_quat[n_quat, 0] = sbg.ekf_quat["yaw acc"]
                        self.status_quat[n_quat]\
                            = sbg.ekf_quat["solution status"]
                        n_quat += 1

                    elif type_ == 8:
                        self.t_nav[n_nav] = sbg.ekf_nav["time stamp"]
                        self.velocity_nav[n_nav, 0] = sbg.ekf_nav["velocity n"]
                        self.velocity_nav[n_nav, 1] = sbg.ekf_nav["velocity e"]
                        self.velocity_nav[n_nav, 2] = sbg.ekf_nav["velocity d"]
                        self.velocity_acc_nav[n_nav, 0] \
                            = sbg.ekf_nav["velocity n acc"]
                        self.velocity_acc_nav[n_nav, 1] \
                            = sbg.ekf_nav["velocity e acc"]
                        self.velocity_acc_nav[n_nav, 2] \
                            = sbg.ekf_nav["velocity d acc"]
                        self.pos_nav[n_nav, 0] \
                            = sbg.ekf_nav["latitude"] * np.pi / 180.
                        self.pos_nav[n_nav, 1] \
                            = sbg.ekf_nav["longitude"] * np.pi / 180.
                        self.pos_nav[n_nav, 2] = - sbg.ekf_nav["altitude" ]
                        self.pos_acc_nav[n_nav, 0] = sbg.ekf_nav["latitude acc"]
                        self.pos_acc_nav[n_nav, 1]\
                            = sbg.ekf_nav["longitude acc"]
                        self.pos_acc_nav[n_nav, 2] = sbg.ekf_nav["altitude acc"]
                        self.geoid_nav[n_nav] = sbg.ekf_nav["undulation"]
                        self.status_nav[n_nav] = sbg.ekf_nav["solution status"]
                        n_nav += 1



                    elif type_ == 9:

                        self.t_ship[n_ship] = sbg.ship_motion["time stamp"]
                        self.heave_period_ship[n_ship]\
                            = sbg.ship_motion["heave period"]
                        self.pos_ship[n_ship, 0] = sbg.ship_motion["surge"]
                        self.pos_ship[n_ship, 1] = sbg.ship_motion["sway"]
                        self.pos_ship[n_ship, 2] = sbg.ship_motion["heave"]
                        self.accel_ship[n_ship, 0] = sbg.ship_motion["accel x"]
                        self.accel_ship[n_ship, 1] = sbg.ship_motion["accel y"]
                        self.accel_ship[n_ship, 2] = sbg.ship_motion["accel z"]
                        self.velocity_ship[n_ship, 0] = sbg.ship_motion["vel x"]
                        self.velocity_ship[n_ship, 1] = sbg.ship_motion["vel y"]
                        self.velocity_ship[n_ship, 2] = sbg.ship_motion["vel z"]
                        self.status_ship[n_ship] = sbg.ship_motion["status"]
                        n_ship += 1

                    elif type_ == 13:

                        self.t_gps_vel[n_gps_vel] = sbg.gps1_vel["time stamp"]
                        self.vel_gps_vel[n_gps_vel, 0] = sbg.gps1_vel["vel n"]
                        self.vel_gps_vel[n_gps_vel, 1] = sbg.gps1_vel["vel e"]
                        self.vel_gps_vel[n_gps_vel, 2] =  sbg.gps1_vel["vel d"]
                        self.vel_acc_gps_vel[n_gps_vel, 0]\
                            = sbg.gps1_vel["vel acc n"]
                        self.vel_acc_gps_vel[n_gps_vel, 1]\
                            = sbg.gps1_vel["vel acc e"]
                        self.vel_acc_gps_vel[n_gps_vel, 2]\
                            = sbg.gps1_vel["vel acc d"]
                        self.tow_gps_vel[n_gps_vel]\
                            = sbg.gps1_vel["gps tow"] / 1000.
                        self.course_gps_vel[n_gps_vel] = sbg.gps1_vel["course"]
                        self.course_acc_gps_vel[n_gps_vel]\
                            = sbg.gps1_vel["course acc"]
                        self.status_gps_vel[n_gps_vel]\
                            =  sbg.gps1_vel["gps vel status"]
                        n_gps_vel += 1



                    elif type_ == 14:

                        self.t_gps[n_gps] = sbg.gps1_pos["time stamp"]
                        self.tow_gps[n_gps] = sbg.gps1_pos["gps tow"] / 1000.
                        self.pos_gps[n_gps, 0] = sbg.gps1_pos["latitude"]\
                            * np.pi / 180.
                        self.pos_gps[n_gps, 1] = sbg.gps1_pos["longitude"]\
                            * np.pi / 180.
                        self.pos_gps[n_gps, 2] = - sbg.gps1_pos["altitude"]
                        self.pos_acc_gps[n_gps, 0] = sbg.gps1_pos["pos acc lat"]
                        self.pos_acc_gps[n_gps, 1] = sbg.gps1_pos["pos acc lon"]
                        self.pos_acc_gps[n_gps, 2] = sbg.gps1_pos["pos acc alt"]
                        self.geoid_gps[n_gps] = sbg.gps1_pos["undulation"]
                        self.status_gps[n_gps] = sbg.gps1_pos["gps pos status"]
                        self.num_sv_gps[n_gps] = sbg.gps1_pos["num sv used"]
                        self.base_station_gps[n_gps]\
                            = sbg.gps1_pos["base station id"]
                        self.diff_age_gps[n_gps]\
                            = sbg.gps1_pos["diff age"] / 100.
                        n_gps += 1


                    elif type_ == 32:

                        self.t_ship_delayed[n_ship_delayed]\
                            = sbg.ship_delayed_motion["time stamp"]
                        self.heave_period_ship_delayed[n_ship_delayed]\
                            = sbg.ship_delayed_motion["heave period"]
                        self.pos_ship_delayed[n_ship_delayed, 0]\
                            = sbg.ship_delayed_motion["surge"]
                        self.pos_ship_delayed[n_ship_delayed, 1]\
                            = sbg.ship_delayed_motion["sway"]
                        self.pos_ship_delayed[n_ship_delayed, 2]\
                            = sbg.ship_delayed_motion["heave"]
                        self.accel_ship_delayed[n_ship_delayed, 0]\
                            = sbg.ship_delayed_motion["accel x"]
                        self.accel_ship_delayed[n_ship_delayed, 1]\
                            = sbg.ship_delayed_motion["accel y"]
                        self.accel_ship_delayed[n_ship_delayed, 2]\
                            = sbg.ship_delayed_motion["accel z"]
                        self.velocity_ship_delayed[n_ship_delayed, 0]\
                            = sbg.ship_delayed_motion["vel x"]
                        self.velocity_ship_delayed[n_ship_delayed, 1]\
                            = sbg.ship_delayed_motion["vel y"]
                        self.velocity_ship_delayed[n_ship_delayed, 2]\
                            = sbg.ship_delayed_motion["vel z"]
                        self.status_ship_delayed[n_ship_delayed]\
                            = sbg.ship_delayed_motion["status"]
                        n_ship_delayed += 1

                    elif type_ == 15:

                        self.t_gps_hdt[n_gps_hdt] = sbg.gps1_hdt["time stamp"]
                        self.tow_gps_hdt[n_gps_hdt]\
                            = sbg.gps1_hdt["gps tow"] / 1000.
                        self.status_gps_hdt[n_gps_hdt]\
                            = sbg.gps1_hdt["hdt status"]
                        self.true_heading_gps_hdt[n_gps_hdt]\
                            = sbg.gps1_hdt["gps true heading"] * np.pi / 180.
                        self.true_heading_acc_gps_hdt[n_gps_hdt]\
                            = sbg.gps1_hdt["gps true heading acc"]\
                            * np.pi / 180.
                        self.pitch_gps_hdt[n_gps_hdt]\
                            = sbg.gps1_hdt["gps pitch"] * np.pi / 180.
                        self.pitch_acc_gps_hdt[n_gps_hdt]\
                            = sbg.gps1_hdt["gps pitch acc"] * np.pi / 180.
                        n_gps_hdt += 1

                    elif type_ == 36:

                        self.t_pressure[n_pressure] = sbg.pressure["time stamp"]
                        self.pressure[n_pressure] = sbg.pressure["pressure"]
                        self.altitude_pressure[n_pressure]\
                            = sbg.pressure["altitude"]
                        self.status_pressure[n_pressure] \
                            = sbg.pressure["altimeter status"]
                        n_pressure += 1

                except (EOFError, struct.error):
                    break

            sbg.close()

            # Passage de l'angle de cap entre 0 et 2pi
            if self.t_euler.size > 0:
                self.att_euler[:,0]\
                    = (self.att_euler[:,0] + 6 * np.pi) % (2 * np.pi)

            # Passage du time stamp à l'heure UTC
            assert self.t_time.shape[0] > 0
            self.t_time[:,0] = osm_utils.unwrap(self.t_time[:,0], 2**32)/1.e6
            a, b = osm_utils.sync(self.t_time[:,1], self.t_time[:,0])
            osm_ui.print_\
                ("regression linéaire sur le temps : a: {:f} b: {:f}"\
                 .format(a, b))

            if self.t_status.size > 0:
                self.t_status = osm_utils.unwrap(self.t_status, 2**32)/1.e6
                self.t_status = a * self.t_status + b

            if self.t_imu.size > 0:
                self.t_imu = osm_utils.unwrap(self.t_imu, 2**32)/1.e6
                self.t_imu = a * self.t_imu + b

            if self.t_mag.size > 0:
                self.t_mag = osm_utils.unwrap(self.t_mag, 2**32)/1.e6
                self.t_mag = a * self.t_mag + b

            if self.t_euler.size > 0:
                self.t_euler = osm_utils.unwrap(self.t_euler, 2**32)/1.e6
                self.t_euler = a * self.t_euler + b

            if self.t_quat.size > 0:
                self.t_quat = osm_utils.unwrap(self.t_quat, 2**32)/1.e6
                self.t_quat = a * self.t_quat + b

            if self.t_nav.size > 0:
                self.t_nav = osm_utils.unwrap(self.t_nav, 2**32)/1.e6
                self.t_nav = a * self.t_nav + b

            if self.t_ship.size > 0:
                self.t_ship = osm_utils.unwrap(self.t_ship, 2**32)/1.e6
                self.t_ship = a * self.t_ship + b

            if self.t_ship_delayed.size > 0:
                self.t_ship_delayed = osm_utils.unwrap\
                    (self.t_ship_delayed, 2**32)/1.e6
                self.t_ship_delayed = a * self.t_ship_delayed + b

            if self.t_gps_vel.size > 0:
                self.t_gps_vel = osm_utils.unwrap(self.t_gps_vel, 2**32)/1.e6
                self.t_gps_vel = a * self.t_gps_vel + b

            if self.t_gps.size > 0:
                self.t_gps = osm_utils.unwrap(self.t_gps, 2**32)/1.e6
                self.t_gps = a * self.t_gps + b

            if self.t_gps_hdt.size > 0:
                self.t_gps_hdt = osm_utils.unwrap(self.t_gps_hdt, 2**32)/1.e6
                self.t_gps_hdt = a * self.t_gps_hdt + b

            if self.t_pressure.size > 0:
                self.t_pressure = osm_utils.unwrap(self.t_pressure, 2**32)/1.e6
                self.t_pressure = a * self.t_pressure + b

            # Décodage des statuts
            self.st_general_main_power   = (self.status[:,0] & 1)
            self.st_general_imu_power    = (self.status[:,0] & 2) >> 1
            self.st_general_gps_power    = (self.status[:,0] & 4) >> 2
            self.st_general_settings     = (self.status[:,0] & 8) >> 3
            self.st_general_temperature  = (self.status[:,0] & 16) >> 4

            self.st_comm_port_a_valid    = (self.status[:,1] & 1)
            self.st_comm_port_b_valid    = (self.status[:,1] & 2) >> 1
            self.st_comm_port_c_valid    = (self.status[:,1] & 4) >> 2
            self.st_comm_port_d_valid    = (self.status[:,1] & 8) >> 3
            self.st_comm_port_e_valid    = (self.status[:,1] & 16) >> 4
            self.st_comm_port_a_rx    = (self.status[:,1] & 32) >> 5
            self.st_comm_port_a_tx    = (self.status[:,1] & 64) >> 6
            self.st_comm_port_b_rx    = (self.status[:,1] & 128) >> 7
            self.st_comm_port_b_tx    = (self.status[:,1] & 256) >> 8
            self.st_comm_port_c_rx    = (self.status[:,1] & 512) >> 9
            self.st_comm_port_c_tx    = (self.status[:,1] & 1024) >> 10
            self.st_comm_port_d_rx    = (self.status[:,1] & 2048) >> 11
            self.st_comm_port_d_tx    = (self.status[:,1] & 4096) >> 12
            self.st_comm_port_e_rx   = (self.status[:,1] & 8192) >> 13
            self.st_comm_port_e_tx    = (self.status[:,1] & 16384) >> 14
            self.st_comm_can_rx    = (self.status[:,1] & 32768) >> 15
            self.st_comm_can_tx    = (self.status[:,1] & 65536) >> 16
            self.st_comm_can_bus   = (self.status[:,1] & 917504) >> 17

            self.st_aiding_gps1_pos    = (self.status[:,2] & 1)
            self.st_aiding_gps1_vel    = (self.status[:,2] & 2) >> 1
            self.st_aiding_gps1_hdt    = (self.status[:,2] & 4) >> 2
            self.st_aiding_gps1_utc    = (self.status[:,2] & 8) >> 3
            self.st_aiding_port_mag    = (self.status[:,2] & 16) >> 4
            self.st_aiding_port_odo    = (self.status[:,2] & 32) >> 5
            self.st_aiding_port_dvl    = (self.status[:,2] & 64) >> 6

            self.st_clock_stable    = (self.status_time & 1)
            self.st_clock_status    = (self.status_time & 30) >> 1
            self.st_clock_utc_sync    = (self.status_time & 32) >> 5
            self.st_clock_utc_status  = (self.status_time & 960) >> 6

            self.st_imu_com    = (self.status_imu & 1)
            self.st_imu_status    = (self.status_imu & 2) >> 1
            self.st_imu_accel_x    = (self.status_imu & 4) >> 2
            self.st_imu_accel_y    = (self.status_imu & 8) >> 3
            self.st_imu_accel_z    = (self.status_imu & 16) >> 4
            self.st_imu_gyro_x    = (self.status_imu & 32) >> 5
            self.st_imu_gyro_y    = (self.status_imu & 64) >> 6
            self.st_imu_gyro_z    = (self.status_imu & 128) >> 7
            self.st_imu_accel_in_range    = (self.status_imu & 256) >> 8
            self.st_imu_gyro_in_range    = (self.status_imu & 512) >> 9

            self.st_euler_solution_mode   = (self.status_euler & 15)
            self.st_euler_attitude_valid    = (self.status_euler & 16) >> 4
            self.st_euler_heading_valid    = (self.status_euler & 32) >> 5
            self.st_euler_velocity_valid    = (self.status_euler & 64) >> 6
            self.st_euler_position_valid    = (self.status_euler & 128) >> 7
            self.st_euler_vert_ref_used    = (self.status_euler & 256) >> 8
            self.st_euler_mag_ref_used    = (self.status_euler & 512) >> 9
            self.st_euler_gps1_vel_used    = (self.status_euler & 1024) >> 10
            self.st_euler_gps1_pos_used    = (self.status_euler & 2048) >> 11
            self.st_euler_gps1_course_used   = (self.status_euler & 4096) >> 12
            self.st_euler_gps1_hdt_used   = (self.status_euler & 8192) >> 13
            self.st_euler_gps2_vel_used    = (self.status_euler & 16384) >> 14
            self.st_euler_gps2_pos_used    = (self.status_euler & 32768) >> 15
            self.st_euler_gps2_course_used   = (self.status_euler & 65536) >> 16
            self.st_euler_gps2_hdt_used   = (self.status_euler & 131072) >> 17
            self.st_euler_odo_used   = (self.status_euler & 262144) >> 18

            self.st_quat_solution_mode   = (self.status_quat & 15)
            self.st_quat_attitude_valid    = (self.status_quat & 16) >> 4
            self.st_quat_heading_valid    = (self.status_quat & 32) >> 5
            self.st_quat_velocity_valid    = (self.status_quat & 64) >> 6
            self.st_quat_position_valid    = (self.status_quat & 128) >> 7
            self.st_quat_vert_ref_used    = (self.status_quat & 256) >> 8
            self.st_quat_mag_ref_used    = (self.status_quat & 512) >> 9
            self.st_quat_gps1_vel_used    = (self.status_quat & 1024) >> 10
            self.st_quat_gps1_pos_used    = (self.status_quat & 2048) >> 11
            self.st_quat_gps1_course_used   = (self.status_quat & 4096) >> 12
            self.st_quat_gps1_hdt_used   = (self.status_quat & 8192) >> 13
            self.st_quat_gps2_vel_used    = (self.status_quat & 16384) >> 14
            self.st_quat_gps2_pos_used    = (self.status_quat & 32768) >> 15
            self.st_quat_gps2_course_used   = (self.status_quat & 65536) >> 16
            self.st_quat_gps2_hdt_used   = (self.status_quat & 131072) >> 17
            self.st_quat_odo_used   = (self.status_quat & 262144) >> 18

            self.st_nav_solution_mode   = (self.status_nav & 15)
            self.st_nav_attitude_valid    = (self.status_nav & 16) >> 4
            self.st_nav_heading_valid    = (self.status_nav & 32) >> 5
            self.st_nav_velocity_valid    = (self.status_nav & 64) >> 6
            self.st_nav_position_valid    = (self.status_nav & 128) >> 7
            self.st_nav_vert_ref_used    = (self.status_nav & 256) >> 8
            self.st_nav_mag_ref_used    = (self.status_nav & 512) >> 9
            self.st_nav_gps1_vel_used    = (self.status_nav & 1024) >> 10
            self.st_nav_gps1_pos_used    = (self.status_nav & 2048) >> 11
            self.st_nav_gps1_course_used   = (self.status_nav & 4096) >> 12
            self.st_nav_gps1_hdt_used   = (self.status_nav & 8192) >> 13
            self.st_nav_gps2_vel_used    = (self.status_nav & 16384) >> 14
            self.st_nav_gps2_pos_used    = (self.status_nav & 32768) >> 15
            self.st_nav_gps2_course_used   = (self.status_nav & 65536) >> 16
            self.st_nav_gps2_hdt_used   = (self.status_nav & 131072) >> 17
            self.st_nav_odo_used   = (self.status_nav & 262144) >> 18

            self.st_ship_heave_valid   = (self.status_ship & 1)
            self.st_ship_heave_vel_aided    = (self.status_ship & 2) >> 1
            self.st_ship_period_available    = (self.status_ship & 4) >> 2
            self.st_ship_period_valid    = (self.status_ship & 8) >> 3

            self.st_ship_delayed_heave_valid   = (self.status_ship_delayed & 1)
            self.st_ship_delayed_heave_vel_aided\
                = (self.status_ship_delayed & 2) >> 1
            self.st_ship_delayed_period_available\
                = (self.status_ship_delayed & 4) >> 2
            self.st_ship_delayed_period_valid\
                = (self.status_ship_delayed & 8) >> 3


            self.st_gps_vel_status   = (self.status_gps_vel & 31)
            self.st_gps_vel_type    = (self.status_gps_vel & 2016) >> 6

            self.st_gps_pos_status   = (self.status_gps & 31)
            self.st_gps_pos_type    = (self.status_gps & 2016) >> 6
            self.st_gps_gps_l1_used   = (self.status_gps & 4096) >> 12
            self.st_gps_gps_l2_used   = (self.status_gps & 8192) >> 13
            self.st_gps_gps_l5_used    = (self.status_gps & 16384) >> 14
            self.st_gps_glo_l1_used    = (self.status_gps & 32768) >> 15
            self.st_gps_glo_l2_used   = (self.status_gps & 65536) >> 16

            self.st_gps_hdt_sol_computed   = (self.status_gps_hdt & 1)
            self.st_gps_hdt_insufficient_obs    = (self.status_gps_hdt & 2) >> 1
            self.st_gps_hdt_internal_error    = (self.status_gps_hdt & 4) >> 2
            self.st_gps_hdt_height_limit    = (self.status_gps_hdt & 8) >> 3

            self.st_pressure_valid   = (self.status_pressure & 1)
            self.st_altitude_valid    = (self.status_pressure & 2) >> 1

            self.st_mag_mag_x    = (self.status_mag & 1)
            self.st_mag_mag_y    = (self.status_mag & 2) >> 1
            self.st_mag_mag_z    = (self.status_mag & 4) >> 2
            self.st_mag_acc_x    = (self.status_mag & 8) >> 3
            self.st_mag_acc_y    = (self.status_mag & 16) >> 4
            self.st_mag_acc_z    = (self.status_mag & 32) >> 5
            self.st_mag_mag_in_range    = (self.status_mag & 64) >> 6
            self.st_mag_accel_in_range    = (self.status_mag & 128) >> 7
            self.st_mag_calibration    = (self.status_mag & 256) >> 8
