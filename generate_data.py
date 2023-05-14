from scipy.special import roots_legendre
import json
import numpy as np
import os


# 生成数据集：[{'data':[],'label':num},{'data':[],'label':num}...}
# 文章使用的是精确解exact solution在单元内的积分
# TODO 使用有限元法数值解求每个cell的平均


class Generate_discon_data():
    def __init__(self, fun1, fun2, data_i=5, init_interval=(0, 2), data_num=(400, 1200), interval=(0, 2*np.pi, 3), N=100, status='train') -> None:
        self.N = N  # 单元个数
        self.data_i=data_i # 输入数据是几个单元一组
        self.xa, self.xb, self.tb = interval
        self.cell_size = (self.xa-self.xb)/N
        self.inita, self.initb = init_interval  # 随机的初值区间范围
        self.num1, self.num2 = data_num
        self.status = status
        self.function1, self.function2 = fun1, fun2
        self.labelfile = self.get_data()
        self.save_path='./train_data'

    def get_data(self):
        # 得到数据,为了数据平衡,每生成一次函数取一个正常区域一个间断区域
        t, xl, xr = np.random.random(3)
        t = t*self.tb
        xl = (self.initb-self.inita)*xl+self.inita
        xr = (self.initb-self.inita)*xr+self.inita
        file1 = []
        while len(file1) < self.num1:
            a = self.function1(t, self.data_i)
            file1 = file1+a
        while len(file1) < self.num2:
            a = self.function2(t, xl, xr, self.data_i)
            file1 = file1+a
        return file1

    def save_file(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(os.path.join(self.save_path, f'{self.status}.json'), 'w', encoding='utf-8') as f:
            json.dump(self.labelfile, f)
        pass


class BURGER():
    # 解函数随机得到一个间断位置一个不间断位置
    # 输出形式[{},{}]
    def __init__(self, gauss_n=5, N=100, interval=(0, 2*np.pi, 6)) -> None:
        self.gauss_n = gauss_n
        self.N = N  # 单元个数
        self.xa, self.xb, self.tb = interval
        self.cell_size = (self.xb-self.xa)/N
        self.xx = np.linspace(self.xa, self.xb, N+1)

    def get_excat_solution(self, t):
        '''输入空间剖分x, 时间t 迭代最大步数N 得到精确解'''
        xc = (self.xx[1:-1]+self.xx[2:])/2  # x_i+1/2
        roots, weights = roots_legendre(self.gauss_n)
        xc_gauss = np.array(
            [roots*self.cell_size*0.5+i for i in xc]).flatten()  # 使用gauss五点计算均值
        omega_new = np.array([xc[0] if i < np.pi else xc[-1]
                              for i in xc_gauss])  # 初值
        for i in range(self.N):
            omega = omega_new
            omega_new = self.iterfunc(omega, xc_gauss, t)
            eps1 = self.u0(omega_new)-self.u0(omega)
            if np.linalg.norm(eps1) <= 0.0001:
                # print("epsilon=", np.linalg.norm(eps1))
                break
        # plt.scatter(xc_gauss, self.u0(omega))
        # plt.show()
        # 五点位置上值加权weight得到精确值
        mean_xc = np.array([sum(self.u0(omega)[i*self.gauss_n:(i+1)*self.gauss_n]*weights)
                           for i in range(len(xc))])
        return xc, mean_xc

    def get_disc_point(self, t, data_i):
        xc, mean_xc = self.get_excat_solution(t)
        h1 = self.cell_size/2
        xx_mid = (self.xb-self.xa)/2
        position = [np.where(xc == i)[0][0] for i in xc if xx_mid >=
                    i-h1 and i+h1 >= xx_mid]  # 由于间断点总在中间所以找中间点
        temp =int(data_i//2)
        disc = {'data': mean_xc[position[0]-temp:position[0]+temp+1], 'label': 1}
        rand_well_posi = np.random.randint(len(xc))
        while rand_well_posi == position[0] or rand_well_posi-temp<0 or rand_well_posi+temp+1>len(xc):
            rand_well_posi = np.random.randint(len(xc))
        well = {'data': mean_xc[rand_well_posi-temp:rand_well_posi+temp+1], 'label': 0}
        ####寻找错误数据
        if len(disc['data'])!=data_i:
            print(disc)
            raise Exception('find it')
        if len(well['data'])!=data_i:
            print(well)
            raise Exception('find it')
        #############
        return [disc, well]

    def get_R_solution(self, t, ul, ur):
        xc = (self.xx[1:-1]+self.xx[2:])/2  # x_i+1/2
        roots, weights = roots_legendre(self.gauss_n)
        xc_gauss = np.array(
            [roots*self.cell_size*0.5+i for i in xc]).flatten()  # 使用gauss五点计算均值
        # 解边值条件
        u = np.zeros_like(xc_gauss)
        u[(xc_gauss - ul * t) < 0] = ul
        u[~((xc_gauss - ul * t) < 0)] = ur
        # 找间断点# 在xc=tur是断点, 但是(离散空间不一定有相等的点
        position =-1
        for i in range(len(xc_gauss)-1):
            if np.abs(u[i] - u[i+1])>1e-5:
                position = int(i//self.gauss_n)

        # plt.scatter(xc_gauss, u)
        # plt.show()
        mean_xc = np.array(
            [sum(u[i*self.gauss_n:(i+1)*self.gauss_n]*weights) for i in range(len(xc))])
        return xc, mean_xc, position

    def get_R_point(self, t, ul, ur, data_i):
        xc, mean_xc, position = self.get_R_solution(t, ul, ur)
        temp =int(data_i//2)
        # 在xc=tur是断点, 但是离散空间不一定有相等的点
        if position-temp>=0 and position+temp+1<=len(xc):# 间断点可能在ur过大时不在区间内,并且确保取到五个点
            disc = {'data': mean_xc[position-temp:position+temp+1], 'label': 1}
        else:
            while position-temp<0 or position+temp+1>len(xc):
                position=np.random.randint(len(xc))
                disc={'data':mean_xc[position-temp:position+temp+1], 'label': 0}
        rand_well_posi = np.random.randint(len(xc))
        while rand_well_posi == position or rand_well_posi-temp<0 or rand_well_posi+temp+1>len(xc):
            rand_well_posi = np.random.randint(len(xc))
        well = {'data': mean_xc[rand_well_posi-temp:rand_well_posi+temp+1], 'label': 0}
        ####寻找错误数据
        if len(disc['data'])!=data_i:
            print(disc)
            raise Exception('find it')
        if len(well['data'])!=data_i:
            print(well)
            raise Exception('find it')
        #############
        return [disc, well]

    def u0(self, x):
        return np.sin(np.pi*x)+0.5  # u(x,0)

    def u0R(self, x, ul, ur):
        return np.array([ul if i < 0 else ur for i in x])

    def iterfunc(self, omega, x, t):
        return omega-(omega+t*np.sin(omega)-x)/(1+t*np.cos(omega))


def main():
    # du/dt + d(u^2/2)/dx = 0
    # solution 1 detect by myself
    test = BURGER()
    # print(test.get_R_point(2, 2, 4))
    # print(test.get_disc_point(2))
    a = Generate_discon_data(test.get_disc_point, test.get_R_point)
    a.get_data()


if __name__ == '__main__':
    main()
