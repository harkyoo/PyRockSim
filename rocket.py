"""
Rocket simulator

Copyright (c) 2016 Kenji Nakakuki
Released under the MIT license
"""

import numpy as np
from numpy import sin, cos, arcsin, pi
import matplotlib as mpl
import matplotlib.pyplot as plt
#import scipy as sp
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap
import quaternion as qt
import environment as env
import coordconv as cc

# Matplotlib의 설정
plt.style.use('ggplot')
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.grid'] = True

# 일반정수 설정
D2R = pi / 180.0
R2D = 180.0 / pi

# Rocket의 재원설정(M-3S)
rocket_settings = {
    'm0': 45247.4,  # [kg] 초기질량
    'Isp': 266,  # [s] Specific Impulse
    'g0': 9.80665,  # [m/s^2] 중력상수
    'FT': 1147000,  # [N] 추력(일정)
    'Tend': 53,  # [s] 로켓 연소 종료 시간
    'Area': 1.41 ** 2 / 4 * pi,  # [m^2] 기준 면적
    'CLa': 3.5,  # [-] 양력 기울기
    'length_GCM': [-9.76, 0, 0],  # [m] R/M 심벌-레버 암 길이
    'length_A': [-1.0, 0, 0],  # [m] 기체 공력 중심-레버 암 길이
    'Ijj': [188106.0, 188106.0, 1839.0],  # [kg*m^2] 관성모멘트
    'IXXdot': 0,  # [kg*m^2/s] 관성모멘트 변화율 X축
    'IYYdot': 0,  # [kg*m^2/s] 관성모멘트 변화율 Y축
    'IZZdot': 0,  # [kg*m^2/s] 관성모멘트 변화율 Z축
    'roll': 0,  # [deg] 초기 롤
    'pitch': 85.0,  # [deg] 초기 피치각
    'yaw': 120.0,  # [deg] 초기 방위각
    'lat0': 31.251008,  # [deg] 발사지점 위도(WGS84)
    'lon0': 131.082301,  # [deg] 발사지점 경도(WGS84)
    'alt0': 194,  # [m] 발사지점 고도(WGS84타원체)
    # CD를 정의하기 위한 Mach 수와 CD 표
    'mach_tbl': np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6,
                          1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]),
    'CD_tbl': np.array([0.28, 0.28, 0.28, 0.29, 0.35, 0.64, 0.67, 0.69,
                        0.66, 0.62, 0.58, 0.55, 0.48, 0.42, 0.38, 0.355,
                        0.33])
}


class RocketSim:
    """로켓 시뮬레이션용 클래스

    [좌표계의 정의]
    사점 중심 관성 좌표계 n : 발사 위치 원점 직교좌표계(North-East-Down)
    로켓 기재좌표계 b : XYZ=전방/우현/하방

    """

    def __init__(self, **kwargs):
        """로켓의 초기값 상태 설정

        (1) 상미분방정식을 사용한 상태값 벡터의 초기값 x0의 설정(14차)
            m0 : 로켓 전체의 초기 질량 [kg] (1x1)
            pos0: 사점 중심 관성좌표계에서의 위치（North-East-Down) [m] (3x1)
            vel0: 사점 중심 관성좌표계에서의 속도（North-East-Down) [m/s] (3x1)
            quat0: 기체좌표계에서 수평좌표계로의 변환을 나타내는 쿼터니온 [-] (4x1)
            omega0: 기최좌표계에서의 기체에 작용하는 각속도 [rad/s] (3x1)
        (2) 로켓의 각종 제원 설정
        (3) 제어 입력+외부 노이즈
        (4) 질점 모형 설정 플래그
        """

        # (1) 상태값 벡터의 초기값 x0의 설정
        pos0 = [0.0, 0.0, 0.0]  # m
        vel0 = [0.0, 0.0, 0.0]  # m/s
        quat0, _ = qt.attitude(kwargs['roll'], kwargs['pitch'],
                               kwargs['yaw'])
        omega0 = [0.0, 0.0, 0.0]  # rad/s
        self.x0 = np.array([kwargs['m0'], *pos0, *vel0, *quat0, *omega0])
        # (2) 로켓의 각종 제원 설정
        self.isp = kwargs['Isp']
        self.g0 = kwargs['g0']
        self.rm_t_end = kwargs['Tend']
        self.cla = kwargs['CLa']
        self.area = kwargs['Area']
        self.length_gcm = kwargs['length_GCM']
        self.length_a = kwargs['length_A']
        self.ixx = kwargs['Ijj'][0]
        self.iyy = kwargs['Ijj'][1]
        self.izz = kwargs['Ijj'][2]
        self.ixxdot = kwargs['IXXdot']
        self.iyydot = kwargs['IYYdot']
        self.izzdot = kwargs['IYYdot']
        self.xr, self.yr, self.zr = \
            cc.blh2ecef(kwargs['lat0'], kwargs['lon0'], kwargs['alt0'])
        self.dcm_x2n_r = cc.dcm_x2n(kwargs['lat0'], kwargs['lon0'])
        self.mach_tbl = kwargs['mach_tbl']
        self.cd_tbl = kwargs['CD_tbl']
        # (3) u : 제어입력 + 외부 노이즈 (현재 추력 일정 제어 입력 및 외부 노이즈는 제로)
        # 추력 [N] 추력 노즐 Pitch 짐벌 각도 [rad] 추력 노즐 Yaw 짐벌 각도 [rad]
        # 롤 제어 토크 [N * m, 수평 좌표계의 바람 벡터 (x, y, z) [m / s]
        self.u = np.array([kwargs['FT'], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # (4) 모멘트를 계산하지 않고 질점 모델로 만들어 버리는 여부 설정
        # 제어 프로그램을 구현하지 않은 경우 다음 매개 변수를 항상 1로 두는 것.
        self.lumped_mass = 1

    def rocket_dynamics(self, x, t, u):
        """로켓의 역학 계산 함수

        인자:
            x: 상태값x (self.x0 참조)
            t: time 시각[s]
            u: 옵션 파라메터
                u[0]: Ft 추력[N]
                u[1]: deltaY 요 짐벌 각[rad]
                u[2]: deltaP 피치 짐벌 각[rad]
                u[3]: Tr 롤 제어 토크[N*m]
                u[4]: VWHx 초기 수평좌표계에서의 바람 벡터 North[m/s]
                u[5]: VWHy 초기 수평좌표계에서의 바람 벡터 East[m/s]
                u[6]: VWHz 초기 수평좌표계에서의 바람 벡터 Down[m/s]
        리턴값:
            dx: 상태값 x의 시간 미분
        """

        # 회전 행렬 dcm (발사 위치 NED -> 현재 위치 NED)를 계산
        # 적분 계산을 발사 위치 NED 좌표계에서 계산하고 있기 때문에 적절히 dcm을 사용해 수정
        px, py, pz = cc.launch2ecef(x[1], x[2], x[3],
                                    self.xr, self.yr, self.zr)
        phi, lam, _ = cc.ecef2blh(px, py, pz)  # output: deg/deg/m
        dcm = cc.dcm_x2n(phi, lam) @ self.dcm_x2n_r.T

        # 대기와 중력의 계산
        # 대기
        ned_now = dcm @ np.array([x[1], x[2], x[3]])
        a, _, rho, _ = env.std_atmosphere(-ned_now[2])
        # 중력 (fgh : 수평 좌표계에있어서의 기체에 걸리는 중력 [N])
        gvec = env.gravity(-ned_now[2], phi * D2R)
        fgh = x[0] * gvec  # NED좌표

        # 추진제의 질량 유량 delta_m [kg / s]
        if t < self.rm_t_end:
            thrust = u[0]
            delta_m = -thrust / self.isp / self.g0
        else:
            thrust = 0
            delta_m = 0

        # 짐벌 각도를 고려한 기체 좌표계의 추력 ftb [N]
        # 짐벌 각도 deltay, deltap [rad]
        deltay = u[1]
        deltap = u[2]
        ftb = thrust * np.array([cos(deltay) * cos(deltap),
                                 -sin(deltay),
                                 -cos(deltay) * sin(deltap)])

        # 공기력
        # 수평 좌표계의 바람 벡터 VWH [m / s] (고급 방향의 분포는없는 것으로한다)
        # 수평 좌표계에서 기체의 대기 속도 벡터 va [m / s]
        va = x[4:7] - u[4:7]  # 최고속도

        # 기체 좌표계에서 수평 좌표계로의 좌표 변환을 나타내는 쿠ォ타니온 quat (quat_B2H)
        quat = x[7:11]  # q_B2H

        # 쿼터니온에서 방향 코사인 행렬 cbn으로 변환
        cbn = qt.rot_coord_mat(quat)

        # 기체 좌표계에서 본 속도 벡터 vab을 추구 속도 좌표계의 정의에서
        # 기체 좌표계에서 본 속도 좌표계의 기저 벡터 [xAB yAB zAB를 구해
        # 속도 좌표계에서 기체 좌표계의 방향 코사인 행렬 DCM_A2B을 요구하고있다.
        # 기체 좌표계에서 공군 fab [N]
        if np.linalg.norm(va) == 0.0:
            xab = np.array([1, 0, 0])  # 기체 좌표계 속도 방향 단위 벡터
            # vab = np.array (0, 0, 0)
        else:
            vab = cbn.T @ va  # 기체 좌표계로 변환 (발사 지점 NED -> 기체 좌표)
            xab = vab / np.linalg.norm(vab)

        yab_sintheta = np.cross(xab, np.array([1, 0, 0]))
        sintheta = np.linalg.norm(yab_sintheta)
        if sintheta < 1.e-10:
            yab = np.array([0, 1, 0])
        else:
            tmp = yab_sintheta / sintheta
            yab = tmp / np.linalg.norm(tmp)

        theta = arcsin(sintheta)
        zab = np.cross(xab, yab)

        # 속도 좌표계에서 본 공군 faa [N]
        f_cd = interp1d(self.mach_tbl, self.cd_tbl, kind='linear', copy=False,
                        fill_value='extrapolate')
        cd = f_cd(np.linalg.norm(va) / a)
        faa = -0.5 * rho * (np.linalg.norm(va) ** 2) * self.area * \
            np.array([cd, 0, self.cla * theta])

        dcm_a2b = np.c_[xab, yab, zab].T  # 전치주의
        fab = dcm_a2b @ faa  # @ 행렬 곱 (Python 3.5 이상)

        # 모멘트 (mom_t : 추력에 의한 것 [Nm] mom_a : 공기력에 의한 것 [Nm])
        if self.lumped_mass == 1:
            moment = [0., 0., 0.]
        else:
            # mom_t = -np.cross(ftb, self.length_gcm)
            # mom_a = -np.cross(fab, self.length_a)
            # moment = mom_t + mom_a + [u[3], 0, 0]
            raise ValueError('Controller is not yet implemented.')

        # 속도 운동 방정식
        ftah = cbn @ (ftb + fab)
        delta_v = (1 / x[0]) * (ftah + fgh)  # 발사 지점 NED 좌표

        # 자세의 운동 방정식
        delta_quat = qt.deltaquat(quat, x[11:14])

        # 각속도의 운동 방정식
        delta_omega = [1 / self.ixx * (moment[0] - self.ixxdot * x[11] -
                                       (self.izz - self.iyy) * x[12] * x[13]),
                       1 / self.iyy * (moment[1] - self.iyydot * x[12] -
                                       (self.ixx - self.izz) * x[13] * x[11]),
                       1 / self.izz * (moment[2] - self.izzdot * x[13] -
                                       (self.iyy - self.ixx) * x[11] * x[12])]

        dx = [delta_m, x[4], x[5], x[6], *delta_v, *delta_quat, *delta_omega]
        return dx

    def odeint_calc(self, t_vec):
        """ODE 솔버를 사용하여 시뮬레이션 계산을 수행하는"""
        #dat, dbg = sp.integrate.odeint(self.rocket_dynamics, self.x0, t_vec,
                                       #(self.u,), rtol=1.e-3, atol=1.e-3,
                                       #full_output=1)
        dat, dbg = odeint(self.rocket_dynamics, self.x0, t_vec,
                                       (self.u,), rtol=1.e-3, atol=1.e-3,
                                       full_output=1)
        return dat, dbg

    def euler_calc(self, t_vec):
        """오일러 방법 적분으로 시뮬레이션 계산"""
        x = self.x0
        dat = np.zeros((t_vec.size, x.size), dtype='float64')
        dat[0, :] = x
        for k in range(t_vec.size - 1):
            dx = self.rocket_dynamics(x, t_vec[k], self.u)
            x += np.array(dx) * 0.002
            dat[k + 1, :] = x
        return dat


def plot_rs(tv, res):
    """시뮬레이션 결과 시각화

    인수
        tv: 시간 벡터 [s]
        res: 시계열 계산 결과 매트릭스 (상태량 x에 대응)
    """

    def plot_pos(t, d):
        """위치 시각화"""
        h = plt.figure(1)
        h.canvas.set_window_title("Fig %2d - 위치（NED）" % h.number)
        plt.subplot(3, 1, 1)
        plt.plot(t, d[:, 1] * 1.e-3)
        plt.xlabel('Time [s]')
        plt.ylabel('Nothern Position [km]')
        plt.subplot(3, 1, 2)
        plt.plot(t, d[:, 2] * 1.e-3)
        plt.xlabel('Time [s]')
        plt.ylabel('Eastern Position [km]')
        plt.subplot(3, 1, 3)
        plt.plot(t, -d[:, 3] * 1.e-3)
        plt.xlabel('Time [s]')
        plt.ylabel('Altitude [km]')

    def plot_vel(t, d):
        """속도 플롯"""
        v_abs = np.zeros(len(d), dtype=float)
        for k in range(len(d)):
            v_abs[k] = np.linalg.norm(d[k, 4:7])
        h = plt.figure(2)
        h.canvas.set_window_title("Fig %2d - 속도（NED）" % h.number)
        plt.subplot(4, 1, 1)
        plt.plot(t, d[:, 4])
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity North [m/s]')
        plt.subplot(4, 1, 2)
        plt.plot(t, d[:, 5])
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity East [m/s]')
        plt.subplot(4, 1, 3)
        plt.plot(t, -d[:, 6])
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity Up [m/s]')
        plt.subplot(4, 1, 4)
        plt.plot(t, v_abs)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity ABS [m/s]')

    def plot_quat(t, d):
        """Quarternion의 플롯"""
        h = plt.figure(3)
        h.canvas.set_window_title("Fig %2d - Quaternion" % h.number)
        plt.subplot(2, 2, 1)
        plt.plot(t, d[:, 7])
        plt.xlabel('Time [s]')
        plt.ylabel('q0')
        plt.subplot(2, 2, 2)
        plt.plot(t, d[:, 8])
        plt.xlabel('Time [s]')
        plt.ylabel('q1')
        plt.subplot(2, 2, 3)
        plt.plot(t, d[:, 9])
        plt.xlabel('Time [s]')
        plt.ylabel('q2')
        plt.subplot(2, 2, 4)
        plt.plot(t, d[:, 10])
        plt.xlabel('Time [s]')
        plt.ylabel('q3')

    def plot_rpy(t, d):
        """각속도의 플롯"""
        h = plt.figure(4)
        h.canvas.set_window_title("Fig %2d - 각속도(Roll/Pitch/Yaw)" % h.number)
        plt.subplot(3, 1, 1)
        plt.plot(t, d[:, 11] * R2D)
        plt.xlabel('Time [s]')
        plt.ylabel('Roll [deg/s]')
        plt.subplot(3, 1, 2)
        plt.plot(t, d[:, 12] * R2D)
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch [deg/s]')
        plt.subplot(3, 1, 3)
        plt.plot(t, d[:, 13] * R2D)
        plt.xlabel('Time [s]')
        plt.ylabel('Yaw [deg/s]')

    def plot_mass(t, d):
        h = plt.figure(5)
        h.canvas.set_window_title("Fig %2d - 질량" % h.number)
        plt.plot(t, d[:, 0])
        plt.xlabel('Time [s]')
        plt.ylabel('Mass [kg]')

    def get_llh(d):
        [xr, yr, zr] = cc.blh2ecef(rocket_settings['lat0'],
                                   rocket_settings['lon0'],
                                   rocket_settings['alt0'])
        x, y, z = cc.launch2ecef(d[:, 1], d[:, 2], d[:, 3], xr, yr, zr)
        llh_out = np.zeros((len(x), 3))
        for k, p in enumerate(zip(x, y, z)):
            llh_out[k, :] = cc.ecef2blh(p[0], p[1], p[2])
        return llh_out

    def plot_llh(t, llh_in):
        """위도・경도・고도의 플롯"""
        h = plt.figure(6)
        h.canvas.set_window_title("Fig %2d - 위도・경도・고도" % h.number)
        plt.subplot(3, 1, 1)
        plt.plot(t, llh_in[:, 0])
        plt.xlabel('Time [s]')
        plt.ylabel('Latitude [deg]')
        plt.subplot(3, 1, 2)
        plt.plot(t, llh_in[:, 1])
        plt.xlabel('Time [s]')
        plt.ylabel('Longitude [deg]')
        plt.subplot(3, 1, 3)
        plt.plot(t, llh_in[:, 2] * 1.e-3)
        plt.xlabel('Time [s]')
        plt.ylabel('Altitude [km]')

    def plot_map(llh_in):
        h = plt.figure(7, figsize=(8, 8))
        h.canvas.set_window_title("Fig %2d - 위치（지도）" % h.number)
        minlon, maxlon = 130, 132.01
        minlat, maxlat = 31, 32.01
        plt.subplot(3, 1, (1, 2))
        m = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat,
                    llcrnrlon=minlon, urcrnrlon=maxlon, lat_ts=30,
                    resolution='h')
        m.drawmeridians(np.arange(minlon, maxlon, 0.5), labels=[0, 0, 0, 1])
        m.drawparallels(np.arange(minlat, maxlat, 0.5), labels=[1, 0, 0, 0])
        m.drawcoastlines()
        m.plot(llh_in[:, 1], llh_in[:, 0], latlon=True)
        plt.subplot(3, 1, 3)
        plt.plot(llh_in[:, 1], llh_in[:, 2] * 1.e-3)
        plt.xlim([minlon, maxlon])
        plt.xlabel('Longitude [deg]')
        plt.ylabel('Altitude [km]')

    # 필요에 따라 플롯을 작성
    plt.close('all')
    plot_pos(tv, res)
    plot_vel(tv, res)
    plot_quat(tv, res)
    plot_rpy(tv, res)
    plot_mass(tv, res)
    llh = get_llh(res)
    plot_llh(tv, llh)
    plot_map(llh)
    plt.show()


if __name__ == "__main__":
    # 로켓 객체를 생성
    rs = RocketSim(**rocket_settings)
    # 계산에 사용하는 시간 벡터
    tvec = np.arange(0, 100, 0.002)
    # 로켓 시뮬레이션 (적분) 실행
    result, deb = rs.odeint_calc(tvec)  # odeint를 사용하여 적분 (속도)
    # result = rs.euler_calc(tvec)  # 오일러 법 적분의 경우
    # 결과 시각화
    plot_rs(tvec, result)
