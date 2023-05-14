import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre


class RKDG():
    def __init__(self, u0, gauss_n=4, interval=(0, 2*np.pi, 11.1), pn=2, N=160, CFL=0.01, RK_n=3) -> None:
        self.xa, self.xb, self.tb = interval  # x的区间长度
        self.pn = pn  # 确定是p几元
        self.N = N  # 空间剖分
        self.RK_n = RK_n  # RK算法中时间的阶数
        self.cell_size = (self.xb-self.xa)/N  # 单元大小
        self.CFL = CFL  # 精度,通量
        self.u0 = u0  # 初值
        self.gauss_n = gauss_n  # 使用高斯积分参数
        
        self.dt=0.1*self.cell_size # 时间微分

    def solve_process(self):
        xx = np.linspace(self.xa, self.xb, self.N + 1)
        xc = (xx[0:-1]+xx[1:])/2  # i+1/2的点
        u,integral_init = self.init_point(xc) #得到初值, (self.N,)(self.pn+1, self.N)
        #时间迭代 4阶龙格库塔法
        step=int(self.tb//self.dt)# RK迭代步数
        dt_lst=self.tb % self.dt #最后一个区间长度
        
        pass

    def Legendre(self, n, x):
        '''
        输入:区间长度a,b和需要第几个基函数pn
        返回:(值,导数)
        '''
        p_1, dp_1, p0, dp0 = 0, 0, 1, 0
        p1, dp1 = x, 1
        if n == -1:
            return p_1, dp_1
        elif n == 0:
            return p0, dp0
        elif n == 1:
            return p1, dp1
        else:
            for i in range(2, n+1):
                p_n = (2*n-1)/n*x*p1-(n-1)*p0/n
                dp_n = (2*n-1)/n*(x*dp1+p1)-(n-1)*dp0/n
                p1, dp1, p0, dp0 = p_n, dp_n, p1, dp1
            return p_n, dp_n

    def standard_Legendre(self, n: int, x, i: int):
        '''
        将基函数变换到第i个单元(从0开始)
        输入单元位置i, 第几个基函数n, 对应位置x'''
        # x_a = self.cell_size*i  # 第i个单元开始位置
        # x_b = self.cell_size*(i+1)
        # x_mid = (x_a+x_b)/2
        h1 = self.cell_size/2
        p_in = np.sqrt((2*n+1)/(2*h1)) * \
            self.Legendre(n, (x)/h1)[0]
        dp_in = np.sqrt((2*n+1)/(2*h1))/h1 * \
            self.Legendre(n, (x)/h1)[1]
        return p_in, dp_in

    def draw_curv(self, x, y):
        plt.scatter(x, y)
        plt.show()
        pass
    
    def init_point(self,xc):
        xc_forinit = np.tile(xc, (self.gauss_n, 1))
        # 用gauss5得到点位置,初值
        roots, weights = roots_legendre(self.gauss_n)
        roots_forinit = np.tile(roots.reshape(self.gauss_n, 1), (1, self.N))
        roots_i = 0.5*self.cell_size*roots_forinit+xc_forinit
        # 计算每个单元内的初值
        integral_init = np.zeros((self.pn+1, self.N))
        for i in range(self.pn+1):
            integral_init[i] = (  # 标准legrendre计算时与单元数i无关
                0.5*self.cell_size*weights*self.standard_Legendre(n=i, x=0.5*self.cell_size*roots, i=0)[0])@self.u0(roots_i)
        temp_Legendre = np.array(
            [self.standard_Legendre(i, 0, 0)[0] for i in range(self.pn+1)])
        Q = temp_Legendre @ integral_init

        self.draw_curv(xc, Q)
        return Q, integral_init
 
def u0(x):
    return np.sin(np.pi*x)+0.5

def fpi(g,x0,eps,N):#迭代法
    '''g迭代函数, x0初始迭代点, eps容差限, N最大迭代次数'''
    x=g(x0)
    n=0#迭代次数
    while np.abs(x-x0)>eps and n<N:
        x0=x
        x=g(x0)
        n+=1
    xc=g(x)
    if n==N:
        print('reached maximum number of iterations')    
    return xc
def main():
    result = RKDG(u0=u0)
    # print(result.standard_Legendre(0, 1, 1))
    result.solve_process()


if __name__ == '__main__':
    main()
