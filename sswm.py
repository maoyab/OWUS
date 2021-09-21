import numpy as np


class SM_C_H(object):
    
    def __init__(self, params, nbins=100, q=1, f_wilt=0.05, f_star=0.95, constraints=True, stress_type='dynamic'):
        self.q = q
        self.nbins = nbins
        self.Mw = 0.018          # [kg mol-1] Molar mass of water        
        self.rho_w = 1000        # [kg m-3] Density of liquid water
        self.g = 9.81            # [m s-2] Gravitational acceleration
        self.hi = 2 * 10 ** - 4  # characteristic amount of interception per unit LAI
        self.k_ext = - np.log(0.05) / 2.5        # extinction coefficient for radiation through canopy
        self.f_star = f_star     # T/Tww at s_star
        self.f_wilt = f_wilt     # T/Tww at s_wilt
        self.dr = 0.0005         # [m] Fine root diameter Manzoni 2014 
         
        [self.T_GS, self.rf_alpha, self.rf_lambda, self.Eo, self.Td,
          self.LAI, self.RAI, self.hc, self.Zm,
          self.Ps0, self.b, self.Ks, self.n, self.s_h, self.s_fc, 
          self.k_xl_max, self.Px50, self.Pg50, 
        ]  = params

        self.phiE = np.exp(-self.k_ext * self.LAI)
        self.delta = self.hi * self.LAI
        self.rf = (self.rf_alpha - self.delta) * self.rf_lambda

        self.K_p_max = self.scaled_K_p_max()
        self.K_sr_max = self.get_K_sr_max()

        self.pi_F,\
        self.pi_R,\
        self.pi_T,\
        self.pi_S = self.get_pi_groups()

        self.beta_ww = self.get_beta_ww()

        self.psi_wilt = self.psi_s_from_beta(self.f_wilt)
        self.s_wilt = self.pot_to_s_BC(self.psi_wilt)
        self.psi_star = self.psi_s_from_beta(self.f_star)
        self.s_star = self.pot_to_s_BC(self.psi_star)

        if self.s_star > self.s_fc:
            s_star_c = self.s_fc - 0.01
            self.beta_ww = (self.s_star - s_star_c) / (self.s_star - self.s_wilt) * self.beta_ww
            self.s_star = s_star_c
            self.psi_star = self.s_to_pot_BC(self.s_star)

        if self.s_h > self.s_wilt:
            self.s_wilt = self.s_h + 0.01 

        self.psi_l_star = self.leaf_water_pot(self.psi_star)
        self.psi_l_th = self.get_psi_l_th()

        if (self.psi_l_star < self.psi_l_th) and constraints:
            self.s_fc = np.nan
            self.s_star = np.nan
            self.s_wilt = np.nan
            self.s_h = np.nan

        self.E_max = self.phiE * self.Eo / (self.n * self.Zm)
        self.T_ww = (1 - self.phiE) * self.beta_ww * self.Eo / (self.n * self.Zm)

        self.bb = 2 * self.b + 4
        self.g_p = self.n * self.Zm / (self.rf_alpha - self.delta)
        self.l_p = self.rf_lambda * np.exp(- self.delta / self.rf_alpha)
        self.w = self.l_p * (self.s_fc - self.s_h) * (self.s_star - self.s_wilt) \
                 / ((self.s_star - self.s_wilt) * self.E_max + (self.s_fc - self.s_h) * self.T_ww)
        self.a = self.l_p * (self.s_fc - self.s_h) / self.E_max
        self.m = self.Ks  / ((self.n * self.Zm) * (np.exp(self.bb * (1 - self.s_fc)) - 1.))

        self.AA = self.get_AA()
        self.AA_p = self.get_AA_p() 

        self.p0 = self.get_p0()
        
        self.et_ok = self.eval_et_ok()
        
        self.tet_part = self.get_mean_t(self.p0) / self.get_mean_et(self.p0)
        self.mean_ef = self.get_mean_et(self.p0) / self.rf
        self.mean_tf = self.get_mean_t(self.p0) / self.rf
        self.mean_di = self.Eo / self.rf

        if self.et_ok == 1:
            self.epsilon_dynamic = self.eval_epsilon_dyn()
            self.epsilon_static = self.eval_epsilon()
        else:
            self.epsilon_dynamic = np.nan
            self.epsilon_static = np.nan
        
        if stress_type=='dynamic':
            self.epsilon = self.epsilon_dynamic
        elif stress_type=='static':
            self.epsilon = self.epsilon_static

        if (self.et_ok == 0) or (self.epsilon <=0) or (self.AA > 1) or (self.beta_ww > 1):
            self.et_ok = 0
            self.beta_ww = np.nan
            self.s_fc = np.nan
            self.s_star = np.nan
            self.s_wilt = np.nan
            self.s_h = np.nan
            self.AA = np.nan
            self.epsilon_dynamic = np.nan
            self.epsilon_static = np.nan
            self.epsilon = np.nan


    def scaled_K_p_max(self):
        K_p_max = self.k_xl_max * self.LAI / self.hc * self.Td / self.rho_w     # [m d-1 MPa-1] maximum plant conductivity
        return K_p_max

    def get_K_sr_max(self):
        K_sr_max = self.Ks * (self.RAI / (self.dr * self.Zm))** 0.5 \
                    * 10**6 / (self.rho_w * self.g)      # [m d-1 MPa-1] soil to root conductivity
        return K_sr_max 

    def soil_root_cond(self, psi_s):
        # exponent (2 * b + 3 - d ) b ~ 2 (d=4 root growth correction factor)
        return self.K_sr_max * (self.Ps0 / psi_s) ** 2

    def get_pi_groups(self): 
        pi_F = -self.Eo / (self.K_p_max * self.Pg50)
        pi_R = self.Pg50 / self.Px50
        pi_T = -(self.K_sr_max * self.Pg50) / self.Eo
        pi_S =  self.Pg50 / self.Ps0
        return  pi_F, pi_R, pi_T, pi_S

    def get_beta_ww(self):
        x = (self.pi_F / 2 + 1) ** 2 - 2 * self.pi_F * self.pi_R 
        beta_ww = 1 - 1 / (2 * self.pi_R) * (1 + self.pi_F / 2 - x ** 0.5)
        return beta_ww

    def pot_to_s_BC(self, psi):
        s = (psi / self.Ps0) ** (- 1 / self.b)
        return s

    def s_to_pot_BC(self, s):
        psi =  self.Ps0 * s ** (- self.b)
        return psi

    def psi_s_from_beta(self, f):
        beta = f * self.beta_ww
        y = 1 / (2 * beta * self.pi_S / self.pi_T)
        xx = 1 - beta
        xxx = self.pi_F * beta / (1 - xx * self.pi_R)
        x = 1 + (2 * xx - xxx) *  4 * beta * self.pi_S * self.pi_S / self.pi_T
        p = y * (x ** 0.5 - 1)
        return p * self.Ps0

    def leaf_water_pot(self, psi_s):
        Ksr = self.soil_root_cond(psi_s)
        T0 = self.Eo
        b1 = T0 + T0 * self.pi_R
        b2 = T0 / self.K_p_max * Ksr \
             - psi_s * self.pi_R * Ksr \
             - 2 * self.Pg50 * Ksr
        c1 = 4 * (T0 * self.Pg50 \
            - psi_s * self.Pg50 * Ksr \
            + T0 / self.K_p_max * self.Pg50 * Ksr)\
            *(2 * self.pi_R * Ksr - T0 / self.Px50)
        c2 = (T0 \
            + T0 * self.pi_R \
            + T0 / self.K_p_max * Ksr \
            - psi_s * self.pi_R * Ksr \
            - 2 * self.Pg50 * Ksr) ** 2
        b3 = (c1 + c2)**0.5
        b = b1 + b2 - b3
        a = T0 / self.Px50 - 2 * self.pi_R * Ksr
        return 1 / a * b

    def get_psi_l_th(self):
        psi_l_th = self.Px50 * (1 + self.psi_star/(2*self.Px50))
        return psi_l_th

    def loss(self, s):
        if s > self.s_fc:
            return self.E_max + self.T_ww + self.m * (np.exp(self.bb * (s - self.s_fc)) - 1)
        elif (s > self.s_star) and (s <= self.s_fc):
            return self.E_max * (s - self.s_h) / (self.s_fc - self.s_h) + self.T_ww
        elif (s > self.s_wilt) and (s <= self.s_star):
            return self.E_max * (s - self.s_h) / (self.s_fc - self.s_h) \
                    + self.T_ww * (s - self.s_wilt) / (self.s_star - self.s_wilt)
        elif (s > self.s_h) and (s <= self.s_wilt):
            return self.E_max * (s - self.s_h) / (self.s_fc - self.s_h)
        else:
            return 0

    def loss_T(self, s):
        if s > self.s_fc:
            return self.T_ww 
        elif (s > self.s_star) and (s <= self.s_fc):
            return self.T_ww
        elif (s > self.s_wilt) and (s <= self.s_star):
            return self.T_ww * (s - self.s_wilt) / (self.s_star - self.s_wilt)
        elif (s > self.s_h) and (s <= self.s_wilt):
            return 0
        else:
            return 0

    def __p0(self, s):
        rho = self.loss(s)

        if (s > self.s_h) and (s <= self.s_wilt):
            p = ((np.exp(- self.g_p * s)) * (s - self.s_h) ** self.a) / rho

        elif (s > self.s_wilt) and (s <= self.s_star):
            p1 = ((np.exp(- self.g_p * s))* (self.s_wilt - self.s_h) ** self.a) / rho
            p2 = ((s - self.s_h)  / (self.s_wilt - self.s_h) \
                   + (self.T_ww * (s - self.s_wilt) * (self.s_fc - self.s_h)) \
                  / (self.E_max * (self.s_star - self.s_wilt) * (self.s_wilt - self.s_h)) ) ** self.w 
            p = p1 * p2

        elif (s > self.s_star) and (s <= self.s_fc):
            p1 = ((np.exp(- self.g_p * s))* (self.s_wilt - self.s_h) ** self.a) / rho
            p2b = ((self.s_star - self.s_h)  / (self.s_wilt - self.s_h) \
                   + (self.T_ww * (self.s_fc - self.s_h)) / (self.E_max * (self.s_wilt - self.s_h)) ) ** self.w
            p3 = ((self.E_max * (s - self.s_h) + self.T_ww * (self.s_fc - self.s_h)) \
                  / (self.E_max * (self.s_star - self.s_h) + self.T_ww * (self.s_fc - self.s_h)) ) ** self.a

            p = p1 * p2b * p3

        elif s > self.s_fc:
            p1 = ((np.exp(- self.g_p * s))* (self.s_wilt - self.s_h) ** self.a) / rho
            p2b = ((self.s_star - self.s_h)  / (self.s_wilt - self.s_h) \
                   + (self.T_ww * (self.s_fc - self.s_h)) / (self.E_max * (self.s_wilt - self.s_h)) ) ** self.w
            p3b = (((self.E_max + self.T_ww) * (self.s_fc - self.s_h)) \
                  / (self.E_max * (self.s_star - self.s_h) + self.T_ww * (self.s_fc - self.s_h)) ) ** self.a
            
            ll = (self.E_max + self.T_ww + self.m * (np.exp(self.bb * (s - self.s_fc)) - 1)) /(self.E_max + self.T_ww)
            ee = -self.bb * (s - self.s_fc)  + np.log(ll)
            p4 = np.exp(self.l_p * ee / (self.bb * (self.m - self.E_max - self.T_ww)))

            p = p1 * p2b * p3b * p4

        else:
            p = 0

        return p

    def get_p0(self):

        s_list = np.linspace(0., 1., (self.nbins + 1))
        p0 = np.array([self.__p0(s) for s in s_list])
        c = np.sum(p0)
        return p0 / c

    def get_mean_t(self, p0):
        Tww = self.T_ww * self.n * self.Zm
        
        def __T_losses(s):
            if (s > self.s_star):
                return Tww 
            elif (s > self.s_wilt) and (s <= self.s_star):
                return Tww  * (s - self.s_wilt) / (self.s_star - self.s_wilt)
            else:
                return 0
        
        s_list = np.linspace(0., 1., len(p0))
        T = np.array([ __T_losses(s) for s in s_list])
        T = T * p0
        return np.sum(T)

    def get_mean_et(self, p0):
        Tww = self.T_ww * self.n * self.Zm
        Emax = self.E_max * self.n * self.Zm

        def __ET_losses(s):
            if s > self.s_fc:
                return Emax + Tww
            
            elif (s > self.s_star) and (s <= self.s_fc):
                return Emax * (s - self.s_h) / (self.s_fc - self.s_h) + Tww
            
            elif (s > self.s_wilt) and (s <= self.s_star):
                return Emax * (s - self.s_h) / (self.s_fc - self.s_h) \
                        + Tww * (s - self.s_wilt) / (self.s_star - self.s_wilt)
            
            elif (s > self.s_h) and (s <= self.s_wilt):
                return Emax * (s - self.s_h) / (self.s_fc - self.s_h)
            
            else:
                return 0
        
        s_list = np.linspace(0., 1., len(p0))
        ET = np.array([ __ET_losses(s) for s in s_list])
        ET = ET * p0
        return np.sum(ET)

    def get_mean_stress(self, p0):
        
        def __Stress(s):
            if (s > self.s_star):
                return 0
            elif (s > self.s_wilt) and (s <= self.s_star):
                return ((self.s_star - s) / (self.s_star - self.s_wilt)) ** self.q
            else:
                return 1
        
        s_list = np.linspace(0., 1., len(p0))
        S = np.array([__Stress(s) for s in s_list])
        S = S * p0
        return np.sum(S)

    def get_mean_beta(self, p0,):
        
        def __bb(s):
            if (s > self.s_star):
                return 1
            elif (s > self.s_wilt) and (s <= self.s_star):
                return ((s - self.s_star) / (self.s_star - self.s_wilt))
            else:
                return 0
        
        s_list = np.linspace(0., 1., len(p0))
        S = np.array([__bb(s) for s in s_list])
        S = S * p0
        return np.sum(S)

    def get_mean_swT(self, p0):
        Tww = self.T_ww * self.n * self.Zm
        
        def __swet(s):
            if (s > self.s_star):
                stress = 0
                return Tww * (1 - self.stress)
            elif (s > self.s_wilt) and (s <= self.s_star):
                stress = ((self.s_star - s) / (self.s_star - self.s_wilt)) ** self.q
                T = Tww * (s - self.s_wilt) / (self.s_star - self.s_wilt)
                return T * (1 - self.stress)
            else:
                stress = 1
                return 0
        
        s_list = np.linspace(0., 1., len(p0))
        S = np.array([ __swet(s) for s in s_list])
        S = S * p0
        return np.sum(S)
        
    def eval_epsilon(self):
        stress = self.get_mean_stress(self.p0)
        transpi = self.get_mean_t(self.p0)
        epsilon = (1 - stress) * transpi / self.rf
        return epsilon

    def get_AA(self):
        AA = (self.s_fc - 0.5 * (self.s_star + self.s_wilt)) \
            / (self.s_fc - self.s_h)
        return AA * self.beta_ww
    
    def get_AA_p(self):
        AA = (-self.s_to_pot_BC(self.s_h)  + 0.5 * (self.psi_star + self.psi_wilt)) \
            / (-self.s_to_pot_BC(self.s_h))
        return AA * self.beta_ww

    def eval_et_ok(self):
        et = self.get_mean_et(self.p0)
        if (et < self.rf_lambda * self.rf_alpha) and (np.isnan(et)==0) and (np.isnan(self.s_star)==0):
            return 1
        else:
            return 0

    def mean_stress_duration(self, p0):
        P0 = np.cumsum(p0)
        l = (len(p0) - 1) 
        P_star = P0[np.int(np.rint(self.s_star * l))]
        p_star = p0[np.int(np.rint(self.s_star * l))]
        rho_star = self.loss(self.s_star)
        return P_star / (p_star * rho_star)

    def mean_stress_frequency(self, p0, T_star):
        l = (len(p0) - 1) 
        p_star = p0[np.int(np.rint(self.s_star * l))]
        rho_star = self.loss(self.s_star)
        return self.T_GS / T_star

    def mean_dynamic_water_stress(self, p0, k=2/3.):
        l = (len(p0) - 1)
        P0 = np.cumsum(p0)
        P_star = P0[np.int(np.rint(self.s_star * l))]
        T_star = self.mean_stress_duration(p0)
        if T_star> self.T_GS:
            T_star = self.T_GS

        n_star = self.mean_stress_frequency(p0, T_star)
        Cstress = self.get_mean_stress(p0)
        e = (n_star)**(-0.5)
        if (Cstress * T_star) <= (k * self.T_GS):
            return ((Cstress * T_star) / (k * self.T_GS)) ** e
        else:
            return 1

    def eval_epsilon_dyn(self):
        stress = self.mean_dynamic_water_stress(self.p0)
        transpi = self.get_mean_t(self.p0)
        epsilon = (1 - stress) * transpi / self.rf
        return epsilon


if __name__ == "__main__":
    pass
