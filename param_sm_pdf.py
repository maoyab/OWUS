import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp, percentileofscore
from sswm import SM_C_H
          

class Inverse_bayesian_fitting(object):
    def __init__(self, s_obs, unknown_params, save_params, p_ranges, epsi=None,  stress_type='dynamic', model_type='A', nbr_sim=20000, burnin=1. / 2., nbins=100):
        self.nbins = nbins
        self.epsi = epsi
        self.stress_type = stress_type
        self.s_obs = s_obs
        self.burnin = burnin
        np.random.seed()
        self.nbr_sim = nbr_sim
        self.model_type = model_type
        self.SM_PDF = SM_C_H
        self.full_param_name_list = ['T_GS', 'rf_alpha', 'rf_lambda', 'Eo', 'Td',
                                      'LAI', 'RAI', 'hc', 'Zm',
                                      'Ps0', 'b', 'Ks', 'n', 's_h', 's_fc', 
                                      'k_xl_max', 'Px50', 'Pg50']

        self.unknown_params = unknown_params
        self.save_params = save_params
        self.p_ranges = p_ranges

    def mcmc_mh(self, theta0_dict):
        accepted = 0.
        (li, theta_i) = self.init_random_model(theta0_dict)
        if (li, theta_i) != (np.nan, np.nan):
            result = [None] * (np.int(self.nbr_sim * (1 - self.burnin)))
            for it in range(self.nbr_sim):
                (acc, l_acc, theta_acc) = self.mcmc_mh_criteria(li, theta_i)
                if it >= self.nbr_sim * self.burnin:
                    theta_list = [getattr(theta_acc, vv) for vv in self.save_params]
                    result[np.int(it - self.nbr_sim * self.burnin)] = [l_acc, theta_list]
                    accepted = accepted + acc
                li, theta_i = l_acc, theta_acc
            return result, accepted / (self.nbr_sim * (1 - self.burnin)) * 100.
        else:
            return [np.nan], 'max init random'

    def init_random_model(self, params0_dict, maxcount=1000):
        params = self.make_random_model(params0_dict)
        while_count = 0
        while self.test_model_consistency(params) < 1 and while_count < maxcount:
            params = self.make_random_model(params0_dict)
            while_count = while_count + 1
        if while_count < maxcount:
            if self.epsi is None:
                smpdf = self.SM_PDF(params, nbins=self.nbins, stress_type=self.stress_type)
                l = self.eval_loglikelihood(smpdf.p0, self.s_obs, params)
            else:
                smpdf = self.SM_PDF(params, nbins=self.nbins, q=self.epsi, stress_type=self.stress_type)
                l = smpdf.epsilon
            if l == 0 or np.isnan(l):
                l = -np.inf
            while_count = 0
            while l == - np.inf and while_count < maxcount:
                while_count = while_count + 1
                params = self.make_random_model(params0_dict)
                while self.test_model_consistency(params) < 1:
                    params = self.make_random_model(params0_dict)
                if self.epsi is None:
                    smpdf = self.SM_PDF(params, nbins=self.nbins, stress_type=self.stress_type)
                    l = self.eval_loglikelihood(smpdf.p0, self.s_obs, params)
                else:
                    smpdf = self.SM_PDF(params, nbins=self.nbins, q=self.epsi, stress_type=self.stress_type)
                    l = smpdf.epsilon
            if while_count < maxcount:
                return l, smpdf
            else:
                print('x', params)
                return np.nan, np.nan
        else:
            print('x', params)
            return np.nan, np.nan

    def test_model_consistency(self, params):
        
        [T_GS, rf_alpha, rf_lambda, Eo, Td,
          LAI, RAI, hc, Zm,
          Ps0, b, Ks, n, s_h, s_fc, 
          k_xl_max, Px50, Pg50, 
        ]  = params

        params_ = [T_GS, rf_alpha, rf_lambda, Eo, Td,
                  LAI, RAI, hc, Zm,
                  -Ps0, b, Ks, n, s_h, s_fc, 
                  k_xl_max, -Px50, -Pg50]

        pi_R = Pg50 / Px50
        K_p_max = k_xl_max * LAI / hc * Td / 1000 
        pi_F = -Eo / (K_p_max * Pg50)

        x = (pi_F / 2 + 1) ** 2 - 2 * pi_F * pi_R 
        beta_ww = 1 - 1 / (2 * pi_R) * (1 + pi_F / 2 - x ** 0.5)

        lnan = len([k for k in params if np.isnan(k)])
        lneg = len([k for k in params_  if k < 0])

        if Px50 > self.p_ranges['Px50'][1] or \
                Px50 < self.p_ranges['Px50'][0] or \
                Pg50 > self.p_ranges['Pg50'][1] or \
                Pg50 < self.p_ranges['Pg50'][0] or \
                k_xl_max > self.p_ranges['k_xl_max'][1] or \
                k_xl_max < self.p_ranges['k_xl_max'][0] or \
                RAI > self.p_ranges['RAI'][1] or \
                RAI < self.p_ranges['RAI'][0] or \
                beta_ww > self.p_ranges['beta_ww'][1] or \
                beta_ww < self.p_ranges['beta_ww'][0] or \
                pi_R > self.p_ranges['pi_R'][1] or \
                pi_R < self.p_ranges['pi_R'][0] or \
                pi_F > self.p_ranges['pi_F'][1] or \
                pi_F < self.p_ranges['pi_F'][0] or \
                pi_R > self.p_ranges['pi_R'][1] or \
                pi_R < self.p_ranges['pi_R'][0] or \
                lneg > 0 or \
                lnan > 0:
            test = 0

        else:
            test = 1
        return test

    def eval_logps(self, p0, s_eval, params):
        def __ev(s):
            p = p0[np.int(np.rint(s * l))]
            return np.log(p)

        l = (len(p0) - 1)
        if s_eval != []:
            return [__ev(s) for s in s_eval]
        else:
            return [-np.inf] 

    def eval_loglikelihood(self, p0, s_eval, params):
        p = self.eval_logps(p0, s_eval, params)
        return np.sum(p)

    def mcmc_mh_criteria(self, li, theta_i):
        lii, theta_ii = self.eval_mh_model(theta_i)
        if theta_ii == []:
            return [0, li, theta_i]
        elif lii > li:
            return [1, lii, theta_ii]
        elif np.random.uniform(0.0, 1.0) < np.exp(lii - li):
            return [1, lii, theta_ii]
        else:
            return [0, li, theta_i]

    def eval_mh_model(self, theta0):
        if self.epsi is None:
            w = 0.02
        else:
            w = 0.2
        params = self.make_mh_model(theta0, w=w)
        if self.test_model_consistency(params) == 1:
            if self.epsi is None:
                smpdf = self.SM_PDF(params, nbins=self.nbins, stress_type=self.stress_type)
                if (smpdf.et_ok == 1) and (np.isnan(smpdf.et_ok) == 0):
                    l = self.eval_loglikelihood(smpdf.p0, self.s_obs, params)
                else:
                    l = np.nan
            else:
                smpdf = self.SM_PDF(params, nbins=self.nbins, q=self.epsi, stress_type=self.stress_type)
                if (smpdf.et_ok == 1) and (np.isnan(smpdf.et_ok) == 0):
                    l = smpdf.epsilon
                else:
                    l = np.nan
            if np.isnan(l) or l == 0:
                l = -np.inf
                smpdf = []
        else:
            l = -np.inf
            smpdf = []
        return l, smpdf

    def make_mh_model(self, theta0, w=0.02):
        params = []
        for vi in self.full_param_name_list:
            if vi in self.unknown_params:
                params.append(np.random.normal(getattr(theta0, vi), w * (self.p_ranges[vi][1] - self.p_ranges[vi][0])))
            else:
                params.append(getattr(theta0, vi))
        return params

    def make_random_model(self, params0):
        params = []
        for vi in self.full_param_name_list:
            if vi in self.unknown_params:
                params.append(np.random.uniform(self.p_ranges[vi][0], self.p_ranges[vi][1]))
            else:
                params.append(params0[vi])
        return params


class Processor(object):
    def __init__(self, model_params_estimate, save_params, epsi=None, model_type='A', nbins=100, stress_type='dynamic'):
        self.model_type = model_type
        self.nbins = nbins
        self.epsi = epsi
        self.model_params_estimate = model_params_estimate
        self.save_params = save_params
        self.stress_type = stress_type
        self.full_param_name_list = ['T_GS', 'rf_alpha', 'rf_lambda', 'Eo', 'Td',
                                      'LAI', 'RAI', 'hc', 'Zm',
                                      'Ps0', 'b', 'Ks', 'n', 's_h', 's_fc', 
                                      'k_xl_max', 'Px50', 'Pg50']

    def get_mcmc_mh_results(self, s_obs, params_dict0, p_ranges, 
                            nbr_sim=20000, num_pl=3, burnin=0.5,
                            efficiency_lim=[0.05, 90]):
        max_num_pl = 3 * num_pl
        pl_results = []
        fail_conv_count = 0
        fail_eff_count = 0
        it_count = 0
        while (len(pl_results) < num_pl) and (it_count < max_num_pl):
            it_count = it_count + 1
            bf = Inverse_bayesian_fitting(s_obs, 
                                          self.model_params_estimate, self.save_params,
                                          p_ranges,
                                          epsi=self.epsi,
                                          stress_type=self.stress_type,
                                          nbr_sim=nbr_sim,
                                          burnin=burnin,
                                          model_type=self.model_type, 
                                          nbins=self.nbins)
            x = bf.mcmc_mh(params_dict0)
            if x[1] != 'max init random':
                result, efficiency = x
                if (efficiency >= efficiency_lim[0]) and (efficiency < efficiency_lim[1]):
                    pl_results.append(x)
                else:
                    fail_eff_count = fail_eff_count + 1
            else:
                print('max init random')
            print('it', it_count, len(pl_results), efficiency, fail_eff_count, num_pl)
        
        
        return pl_results, it_count, fail_conv_count, fail_eff_count

    def check_int_convergeance(self, pl_results0, it, gr_th=1.1, max_num_pl=10):
        pl_results, efficiency = zip(*pl_results0)
        loglikelihood = [list(zip(*r))[0] for r in pl_results]

        estimated_params = [zip(*list(zip(*r))[1]) for r in pl_results]
        estimated_params = zip(*estimated_params)

        gr_lk = self.gelman_rubin_diagnostic([x for x in loglikelihood])
        lk_mean = [np.mean(x) for x in loglikelihood]
        gr_list = []
        for p, est_ in zip(self.model_params_estimate, estimated_params):
            gr = self.gelman_rubin_diagnostic([x for x in est_])
            gr_list.append(gr)

        pl_results_r = pl_results0
        it_conv = 0
        return pl_results_r, it_conv

    def gelman_rubin_diagnostic(self, results):
        k = np.float(len(results))
        n = np.float(len(results[0]))
        means = [np.mean(r) for r in results]
        all_mean = np.mean(means)
        b = n / (k - 1) * \
            np.sum([(mi - all_mean) ** 2 for mi in means])
        w = 1. / (k * (n-1)) * \
            np.sum([(ri - mi) ** 2 for (result, mi) in zip(results, means) for ri in result])
        return ((w * (n - 1) / n + b / n) / w) ** 0.5

    def process_raw_results(self, result_dict, pl_results, p_ranges, outfile_format='short'):
        def __nse(obs, mod):
            mo = np.mean(obs)
            a = np.sum([(mi - oi) ** 2 for mi, oi in zip(mod, obs)])
            b = np.sum([(oi - mo) ** 2 for oi in obs])
            return 1 - a / b
        SM_PDF = SM_C_H

        pl_results, efficiency = zip(*pl_results)
        result_dict['efficiency_estimates'] = efficiency
        result_dict['efficiency'] = np.nanmean(efficiency)
        loglikelihood = [list(zip(*r))[0] for r in pl_results]

        estimated_params = [zip(*list(zip(*r))[1]) for r in pl_results]
        estimated_params = zip(*estimated_params)
        result_dict['loglikelihood_estimates'] = loglikelihood
        result_dict['loglikelihood'] = np.median([np.median(llk) for llk in loglikelihood])

        gr = self.gelman_rubin_diagnostic([x for x in loglikelihood])
        result_dict['loglikelihood_grd'] = gr

        for p, est_ in zip(self.save_params, estimated_params):
            gr = self.gelman_rubin_diagnostic([x for x in est_])
            result_dict['%s_grd' % p] = gr
            eflat = np.array(est_).flatten()
            lflat = np.array(loglikelihood).flatten()
            e_maxl = eflat[list(lflat).index(np.nanmax(lflat))]
            result_dict['%s_maxmaxlike' % p] = e_maxl
            result_dict['%s_median' % p] = np.median(eflat)
            result_dict['%s_mean' % p] = np.mean(eflat)
            result_dict['%s_std' % p] = np.std(eflat)
            result_dict['%s_estimates' % p] = est_

            if self.epsi is None:
                result_dict['%s' % p] = np.median(eflat)
            else:
                result_dict['%s' % p] = e_maxl

        theta = [result_dict[vi] for vi in self.full_param_name_list]

        smpdf = SM_PDF(theta, nbins=self.nbins, stress_type=self.stress_type)

        p_fitted_norm = smpdf.p0
        cdf = np.cumsum(p_fitted_norm)
        cdf_m_n = cdf / np.max(cdf)

        f = interp1d(cdf, np.linspace(0, 1, len(p_fitted_norm)))
        random_p = [np.random.uniform(0, 1) for r in range(365)]
        fit_s = np.array(f(random_p))
        (kstat, kstatp) = ks_2samp(result_dict['s_obs'], fit_s)

        q_obs = [percentileofscore(result_dict['s_obs'], s_obs_i, 'weak') / 100. for s_obs_i in result_dict['s_obs']]
        s_mod = [(np.abs(cdf_m_n - qi)).argmin() / np.float(len(p_fitted_norm) - 1) for qi in q_obs]
        result_dict['NSE_O'] = __nse(result_dict['s_obs'], s_mod)

        s_mod_2 = [(np.abs(cdf_m_n - qi / 365.)).argmin() / np.float(len(p_fitted_norm) - 1) for qi in range(1, 365)]
        obs_l2 = [np.percentile(result_dict['s_obs'], qi / 365. * 100) for qi in range(1, 365)]
        result_dict['NSE'] = __nse(obs_l2, s_mod_2)

        result_dict['ks_stat'] = kstat
        result_dict['ks_stat_p'] = kstatp
        result_dict['model_s_bias'] = (np.mean(fit_s) - np.mean(result_dict['s_obs'])) / np.mean(result_dict['s_obs'])

        if outfile_format == 'short':
            result_dict['s_obs'] = []
            for k in result_dict.keys():
                if k.endswith('estimates'):
                    result_dict[k] = []

        return result_dict


if __name__ == "__main__":
    pass
