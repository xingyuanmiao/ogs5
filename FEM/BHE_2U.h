/**
* \file BHE_2U.h
* 2014/06/04 HS inital implementation
* class of borehole heat exchanger with 2 U-tube
*
*/

#ifndef BHE_2U_H
#define BHE_2U_H

#include "BHEAbstract.h"
#include "makros.h"
#include "tools.h"

namespace BHE  // namespace of borehole heat exchanger
{

	class BHE_2U : public BHEAbstract
	{
	public:
		/**
		  * constructor
		  */
        BHE_2U(const std::string name               /* name of the BHE */,
               BHE::BHE_BOUNDARY_TYPE bound_type    /* type of BHE boundary */,
			   bool   if_use_ext_Ra_Rb            /* whether Ra and Rb values are used */,
			   bool user_defined_R_vals           /* when user defined R values are used*/,
               double my_L          = 100           /* length/depth of the BHE */,
			   double my_D          = 0.013         /* diameter of the BHE */, 
			   double my_Qr         = 21.86 / 86400 /* total refrigerant flow discharge of BHE */,
			   double my_r_inner    = 0.016         /* inner radius of the pipline */,
			   double my_r_outer    = 0.016         /* outer radius of the pipline */,
			   double my_b_in       = 0.0029        /* pipe-in wall thickness*/,
			   double my_b_out      = 0.0029        /* pipe-out wall thickness*/,
			   double my_mu_r       = 0.00054741    /* dynamic viscosity of the refrigerant */,
			   double my_rho_r      = 988.1         /* density of the refrigerant */,
			   double my_alpha_L    = 1.0e-4        /* longitudinal dispersivity of the refrigerant in the pipeline */,
               double my_heat_cap_r = 4180          /* specific heat capacity of the refrigerant */,
               double my_rho_g      = 2190          /* density of the grout */,
			   double my_porosity_g = 0.5           /* porosity of the grout */,
               double my_heat_cap_g = 1000          /* specific heat capacity of the grout */,
			   double my_lambda_r   = 0.6405        /* thermal conductivity of the refrigerant */,
			   double my_lambda_p   = 0.38          /* thermal conductivity of the pipe wall */,
			   double my_lambda_g   = 2.3           /* thermal conductivity of the grout */, 
			   double my_omega      = 0.04242       /* pipe distance */,
               double my_power_in_watt = 0.0        /* injected or extracted power */,
               std::size_t my_power_curve_idx = -1  /* index of the power curve*/,
               double my_delta_T_val = 0.0          /* Temperature difference btw inflow and outflow temperature */,
               double my_ext_Ra = 0.0               /* external defined borehole internal thermal resistance */,
               double my_ext_Rb = 0.0               /* external defined borehole thermal resistance */,
			   double my_ext_Rfig = 0.0           /* external defined borehole thermal resistance */,
			   double my_ext_Rfog = 0.0           /* external defined borehole thermal resistance */,
			   double my_ext_Rgg1 = 0.0           /* external defined borehole thermal resistance */,
			   double my_ext_Rgg2 = 0.0           /* external defined borehole thermal resistance */,
			   double my_ext_Rgs = 0.0           /* external defined borehole thermal resistance */,
			   int my_bhe_heating_cop_curve_idx = -1      /* heating cop curve index */,
			   int my_bhe_cooling_cop_curve_idx = -1      /* cooling cop curve index */,
			   bool if_flowrate_curve = false     /* whether flowrate curve is used*/,
			   int my_flowrate_curve_idx = -1     /* flow rate curve index*/,
			   double my_threshold = 0.0            /* Threshold Q value for switching off the BHE when using Q_Curve_fixed_dT B.C.*/,
			   BHE_DISCHARGE_TYPE type = BHE::BHE_DISCHARGE_TYPE_PARALLEL) 
			   : BHEAbstract(BHE::BHE_TYPE_2U, name, bound_type, if_use_ext_Ra_Rb, user_defined_R_vals, my_bhe_heating_cop_curve_idx, my_bhe_cooling_cop_curve_idx),
			_discharge_type(type)
		{
			_u = Eigen::Vector4d::Zero();
			_Nu = Eigen::Vector4d::Zero(); 

			L = my_L;
			D = my_D; 
			Q_r = my_Qr;
			r_inner = my_r_inner;
			r_outer = my_r_outer; 
			b_in = my_b_in; 
			b_out = my_b_out; 
			mu_r = my_mu_r; 
			rho_r = my_rho_r; 
			alpha_L = my_alpha_L;
			heat_cap_r = my_heat_cap_r; 
            rho_g = my_rho_g; 
            heat_cap_g = my_heat_cap_g; 
			porosity_g = my_porosity_g;
			lambda_r = my_lambda_r;
			lambda_p = my_lambda_p; 
			lambda_g = my_lambda_g; 
			omega = my_omega; 
            power_in_watt_val = my_power_in_watt; 
            power_in_watt_curve_idx = my_power_curve_idx;
            delta_T_val = my_delta_T_val; 
			threshold = my_threshold;
            if (if_use_ext_Ra_Rb)
            {
                use_ext_therm_resis = true;
                ext_Ra = my_ext_Ra;
                ext_Rb = my_ext_Rb;
            }
			if (user_defined_R_vals)
			{
				user_defined_therm_resis = true;
				ext_Rfig = my_ext_Rfig;
				ext_Rfog = my_ext_Rfog;
				ext_Rgg1 = my_ext_Rgg1;
				ext_Rgg2 = my_ext_Rgg2;
				ext_Rgs = my_ext_Rgs;
			}
			if (if_flowrate_curve)
			{
				use_flowrate_curve = true;
				flowrate_curve_idx = my_flowrate_curve_idx;
			}

			S_i  = PI * 2.0 * r_outer;
			S_o  = PI * 2.0 * r_outer; 
			S_g1 = 0.5 * D; 
			S_g2 = D;
			S_gs = 0.25 * PI * D; 

            // cross section area calculation
            CSA_i = PI * r_inner * r_inner;
            CSA_o = PI * r_inner * r_inner;
            CSA_g1 = PI * (0.0625 * D * D - r_outer * r_outer); // one fourth of the crosssection minus the crossection of pipe
			CSA_g2 = PI * (0.0625 * D * D - r_outer * r_outer); // one fourth of the crosssection minus the crossection of pipe

			// initialization calculation
			initialize(); 
		}; 

		/**
		  * return the number of unknowns needed for 2U BHE
		  */
		std::size_t get_n_unknowns() { return 8; }

        /**
          * return the number of boundary heat exchange terms for this BHE
          * abstract function, need to be realized.
          */
        std::size_t get_n_heat_exchange_terms() { return 5; }

        /**
          * set the global index of T_in and T_out
          */
        void set_T_in_out_global_idx(std::size_t start_idx);

		double set_BC(double T_in, double current_time)
		{
			return 0;
		}

		double get_flowrate()
		{
			return Q_r;
		}

		void update_flowrate_from_curve(double current_time)
		{
			if (use_flowrate_curve)
			{
				int flag_valid = false;
				double Q_r_tmp = GetCurveValue(flowrate_curve_idx, 0, current_time, &flag_valid);
				update_flow_rate(Q_r_tmp);
			}
		};

        /**
          * set the global index of T_in and T_out at the bottom of the BHE
          */
        void set_T_in_out_bottom_global_idx(std::size_t dof_bhe);

		/**
		  * return the thermal resistance for the inlet pipline
		  * idx is the index, when 2U case,
		  * 0 - the first u-tube
		  * 1 - the second u-tube
		  */
		double get_thermal_resistance_fig(std::size_t idx);

		/**
		  * return the thermal resistance for the outlet pipline
		  * idx is the index, when 2U case,
		  * 0 - the first u-tube
		  * 1 - the second u-tube
		  */
		double get_thermal_resistance_fog(std::size_t idx);

		/**
          * return the thermal resistance
		  */
		double get_thermal_resistance(std::size_t idx);

		/**
		  * calculate thermal resistance
		  */
		void calc_thermal_resistances();
		
		/**
		  * Nusselt number calculation
		  */
		void calc_Nu();

		/**
		  * Renolds number calculation
		  */
		void calc_Re();

		/**
		  * Prandtl number calculation
		  */
		void calc_Pr();

		/**
		  * flow velocity inside the pipeline
		  */
		void calc_u();

		/**
		  * calculate heat transfer coefficient
		  */
		void calc_heat_transfer_coefficients();

        /**
          * return the coeff of mass matrix,
          * depending on the index of unknown.
          */
        double get_mass_coeff(std::size_t idx_unknown);

        /**
          * return the coeff of laplace matrix,
          * depending on the index of unknown.
          */
        void get_laplace_matrix(std::size_t idx_unknown, Eigen::MatrixXd & mat_laplace);

        /**
          * return the coeff of advection matrix,
          * depending on the index of unknown.
          */
        void get_advection_vector(std::size_t idx_unknown, Eigen::VectorXd & vec_advection);

        /**
          * return the coeff of boundary heat exchange matrix,
          * depending on the index of unknown.
          */
        double get_boundary_heat_exchange_coeff(std::size_t idx_unknown);

        /**
          * return the shift index based on primary variable value
          */
        int get_loc_shift_by_pv(FiniteElement::PrimaryVariable pv_name);

        /**
          * return the number of grout zones in this BHE.
          */
        std::size_t get_n_grout_zones(void) { return 4; };

        /**
          * return the inflow temperature based on outflow temperature and fixed power.
          */
        double get_Tin_by_Tout(double T_out, double current_time);

        /**
          * required by eigen library, 
          * to make sure the dynamically allocated class has 
          * aligned "operator new"
          */
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		
	private:
		/**
		  * thermal resistances 
		  */
		double _R_fig, _R_fog; 

		/**
		  * thermal resistances due to advective flow of refrigerant in the pipes
		  */
		double _R_adv_i1, _R_adv_i2, _R_adv_o1, _R_adv_o2;

		/**
          * thermal resistances due to the pipe wall material 
		  */
		double _R_con_a_i1, _R_con_a_i2, _R_con_a_o1, _R_con_a_o2;

		/**
		  * thermal resistances due to the grout transition
		  */
		double _R_con_b; 

		/**
		  * thermal resistances of the grout
		  */
		double _R_g;

		/**
		  * thermal resistances of the grout soil exchange
		  */
		double _R_gs; 

		/**
		  * thermal resistances due to inter-grout exchange
		  */
		double _R_gg_1, _R_gg_2;

		/**
		  * heat transfer coefficients
		  */
		double _PHI_fig, _PHI_fog, _PHI_gg_1, _PHI_gg_2, _PHI_gs;

		/**
		  * Reynolds number
		  */
		double _Re;

		/**
		  * Prandtl number
		  */
		double _Pr;

		/**
		  * Nusselt number
		  */
		Eigen::Vector4d _Nu;

		/**
		  * flow velocity inside the pipeline
		  */
		Eigen::Vector4d _u;

		/**
		  * discharge type of the 2U BHE
		  */
		const BHE_DISCHARGE_TYPE _discharge_type; 

		/**
          * pipe distance
		  */
		double omega; 

		/**
		  * specific exchange surfaces S
		  */
		double S_i, S_o, S_g1, S_g2, S_gs;
        /**
          * cross section area
          */
        double CSA_i, CSA_o, CSA_g1, CSA_g2;
	};


}  // end of namespace

#endif
