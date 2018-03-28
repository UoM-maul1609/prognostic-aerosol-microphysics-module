	!>@author
	!>Paul Connolly, The University of Manchester
	!>@brief
	!>code to allocate arrays, and call activation 
	module sub
	use nrtype1
	implicit none
		real(sp), parameter :: grav=9.8_sp, lv=2.5e6_sp,cp=1005._sp,molw_air=29.e-3_sp,     &
									   r_air=287._sp,r_vap=461._sp,r_gas=8.314_sp,molw_vap=18.e-3_sp, &
									   eps=r_air/r_vap,kappa=r_air/cp,rhow=1000._sp,sigma=72.e-3_sp
									   ! private
		real(sp) :: rhinit,tinit,pinit,w,  &
							mass_dummy,density_dummy,n_dummy,sig_dummy,d_dummy, &
							tcb, pcb,xmin,a,smax, &
							alpha_sup, sigma_sup, g, chi, sd_dummy, s,c0   ! private
		! size n_mode
		real(sp), allocatable, dimension(:) :: n_aer1, sig_aer1, d_aer1, &
										n_aer, sig_aer, d_aer, d_aer_new, sgi, &
										density_final,mass_initial, & !public
									   mass_final,sd,b,sm,eta,f1,f2 ! private
							  
		! size n_mode
		real(sp), allocatable, dimension(:) :: density_core, & 
								  molw_core, & 
								  nu_core,act_frac,act_frac1, act_frac2 , &
								  density_core1, & 
								  molw_core1, & 
								  nu_core1
		! size n_sv
		real(sp), allocatable, dimension(:) :: molw_org, r_org, log_c_star, cstar, &
												org_content, org_content1, &
											  density_org,nu_org,mass_org_condensed, &
											  delta_h_vap, epsilon1, c_ions, &
											  molw_org1, log_c_star1, &
											  density_org1,nu_org1,&
											  delta_h_vap1
											  
		real(sp), dimension(6) :: c1	! private
		logical(lgt) :: check			! private
		integer(i4b) :: n_mode_s		! private
		real(sp) :: p_test, t_test, w_test, a_eq_7, b_eq_7 ! public
		
		integer(i4b) :: n_mode, n_sv, method_flag, giant_flag, sv_flag ! 1=abdul-razzak, ghan; 2=fountoukis and nenes; 3=fountoukis and nenes with quad
	
	private 
	public :: ctmm_activation, allocate_arrays, initialise_arrays, &
		n_mode, n_sv, method_flag, giant_flag, sv_flag, &
		p_test, t_test, w_test, a_eq_7, b_eq_7, n_aer1, d_aer1, sig_aer1, &
		org_content1, molw_org1, log_c_star1, density_org1, nu_org1, delta_h_vap1, &
		molw_core1, density_core1, nu_core1, act_frac1
				
	contains
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculate the activated fraction of a lognormally distributed
	!>aerosol distribution including condensation of semi-volatile organics.
	!>code follows a paper by Connolly, Topping, Malavelle and Mcfiggans (in ACP 2014)
	!>parameterisation developed at university of manchester
	!>@param[in] n_modes1, n_sv1 : number of aerosol and volatility modes
	!>@param[in] sv_flag: flag for cocondensation
	!>@param[inout] n_aer, d_aer, sig_aer, molw_core, density_core, nu_core
	!>@param[in] org_content1: amount of organic in volatility bins
	!>@param[in] molw_org1: molecular weight in volatility bins
	!>@param[in] density_org1: density of organic in volatility bins
	!>@param[in] delta_h_vap1: enthalpy change in volatility bins
	!>@param[in] nu_org1: van hoff factor in volatility bins
	!>@param[in] log_c_star1: volatility bins
	!>@param[in] w1, t1, p1, a, b: vertical wind, temperature, pressure + params in ARG
	!>@param[inout] act_frac1: activated fraction in each mode
	subroutine ctmm_activation(n_modes1,n_sv1,sv_flag, n_aer1,d_aer1,sig_aer1,molw_core1, &
							   density_core1, nu_core1, org_content1, &
							   molw_org1, density_org1, delta_h_vap1, nu_org1,  &
                               log_c_star1, &
                               w1, t1,p1,a_arg,b_arg, &
							   act_frac1 )

		use nrtype1
		use nr1, only : zbrent,qsimp,qromb,brent,midpnt
		implicit none 
			  real(sp), dimension(:), intent(inout) :: n_aer1,d_aer1, sig_aer1, molw_core1, &
													density_core1, nu_core1
			  real(sp), dimension(:), intent(in) :: org_content1  , molw_org1, &
			  							density_org1, delta_h_vap1, nu_org1, log_c_star1                               
			  real(sp), intent(in) :: w1,t1,p1, a_arg, b_arg
			  integer, intent(in) :: n_modes1, n_sv1, sv_flag
			  real(sp), dimension(:), intent(inout) :: act_frac1

		integer(i4b):: i
		
		n_mode_s=n_modes1
		n_aer=n_aer1
		d_aer=d_aer1
		sig_aer=sig_aer1
		molw_core=molw_core1
		density_core=density_core1
		nu_core=nu_core1
		

		org_content=org_content1*1e-9_sp/(p1/r_air/t1) ! kg/kg
		molw_org=molw_org1
		density_org=density_org1
		delta_h_vap=delta_h_vap1
		nu_org=nu_org1
		log_c_star=log_c_star1

		
		if(sv_flag.eq.1) then

			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			! Find how much semi-volatile is condensed
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			call solve_semivolatiles(n_modes1,n_sv1, &
					org_content, log_c_star, delta_h_vap, nu_org, molw_org, &
					mass_initial, nu_core, molw_core,rhinit, t1, &
					mass_org_condensed)
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			! distribute mass in proportion to number - this is wrong - Crooks has a 
			! better method
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			mass_final=mass_initial+sum(mass_org_condensed)* &
									n_aer/sum(n_aer(1:n_modes1)) 
									! final mass after co-condensation
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			! calculate the new density - this is wrong - Crooks has a 
			! better method
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			density_final=mass_initial/density_core+sum(mass_org_condensed/density_org) * &
													n_aer/sum(n_aer(1:n_modes1))
			density_final=mass_final/density_final
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			! calculate the arithmetic standard deviations                               !
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			sd=exp(log(d_aer)+0.5_sp*sig_aer)*sqrt(exp(sig_aer**2)-1._sp)
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			! note that for multiple modes, assume sd remains constant for all modes and 
			! shift each mode by the same amount in diameter space, such that the total 
			! mass constraint is satisfied. (see Connolly et al, 2014 acp))
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			! now calculate the new median diameter                                      !
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			do i=1,n_modes1
				density_dummy=density_final(i)
				mass_dummy=mass_final(i)
				n_dummy=n_aer(i)
				sd_dummy=sd(i)
				xmin=brent(d_aer(i),d_aer(i)*1.01_sp,2000.e-9_sp, &
							mass_integrate,1.e-10_sp,d_aer_new(i))
			enddo
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			! now calculate the new geometric standard deviation                         !
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			d_aer=d_aer_new
			do i=1,n_modes1
				d_dummy=d_aer_new(i)
				sd_dummy=sd(i)
				sig_aer(i)=zbrent(find_sig_aer,1e-9_sp,2e0_sp,1.e-6_sp) ! 1.d-30
			enddo
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		else if(sv_flag.eq.0) then
			mass_org_condensed=0._sp
			mass_final=mass_initial
			density_final=density_core
		endif
		
		
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		! now calculate the activated fraction                                           !
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		if(method_flag.eq.1) then
			a=2._sp*sigma*molw_vap/(rhow*r_gas*tcb)             ! eq 5: abdul-razzak, ghan
			! above, 2 should be 4 for fountoukis and nenes
  
			b=density_final/ &
			  ( (molw_core*mass_initial/nu_core+ & 
			  sum(molw_org*mass_org_condensed/nu_org)* &
			    mass_initial/sum(mass_initial(1:n_modes1))) / &
			  mass_final)/(rhow/molw_vap)                       ! eq 6: abdul-razzak, ghan
			sm=2._sp/sqrt(b)*(a/(3._sp*d_aer/2._sp))**1.5_sp    ! eq 8: abdul-razzak, ghan 
																! or 9: of 2000 paper

			alpha_sup=grav*molw_vap*lv/(cp*r_gas*tcb**2)- &
					  grav*molw_air/(r_gas*tcb)                ! eq 11: abdul-razzak, ghan

			sigma_sup=r_gas*tcb/(svp(tcb)*molw_vap)+ &
					  molw_vap*lv**2/(cp*pcb*molw_air*tcb)     ! eq 12: abdul-razzak, ghan

			g=rhow*r_gas*tcb/(svp(tcb)*dd(tcb,pcb)*molw_vap) + &
			  lv*rhow/(ka(tcb)*tcb)* &
			  (lv*molw_vap/(r_gas*tcb)-1._sp)                     
			g=1._sp/g                                          ! eq 16: abdul-razzak, ghan

			eta=(alpha_sup*w/g)**1.5_sp/ &
				(2._sp*pi*rhow*sigma_sup*n_aer)                ! eq 22: abdul-razzak, ghan
															   ! or 11: of 2000 paper

			chi=(2._sp/3._sp)*(alpha_sup*w/g)**0.5_sp*a       ! eq 23: abdul-razzak, ghan
															   ! or 10: of 2000 paper
 
			! f1=1.5_sp*exp(2.25_sp*sig_aer**2)                ! eq 28: abdul-razzak, ghan
			f1=a_arg*exp(b_arg*sig_aer**2)                  ! or 7 : of 2000 paper 
															   ! a=0.5, b=2.5
										
		    f2=1._sp+0.25_sp*sig_aer                          ! eq 29: abdul-razzak, ghan
															   ! or 8 : of 2000 paper
 
			! act_frac=0.5_sp*erfc(log(f1*(chi/eta)**1.5_sp &
			!          +f2*(sm**2/(eta+3_sp*chi))**0.75_sp) / &
			!          (3._sp*sqrt(2_sp)*sig_aer))             ! eq 30: abdul-razzak, ghan

			smax=sum(1._sp/sm(1:n_modes1)**2* &
			      (f1(1:n_modes1)*(chi/eta(1:n_modes1))**1.5_sp+ &
				  f2(1:n_modes1)*(sm(1:n_modes1)**2._sp / &
				  (eta(1:n_modes1)+3._sp*chi))**0.75_sp ))**0.5_sp
			smax=1._sp/smax					               ! eq 6: of 2000 paper
													
			! smax=(f1*(chi/eta)**1.5_sp+ &
			!       f2*(sm**2/eta)**0.75)**0.5
			! smax=sm/smax                                     ! eq 31: abdul-razzak, ghan

			act_frac1=1._sp/sum(n_aer(1:n_modes1))* &
			          sum(n_aer(1:n_modes1)*5.e-1_sp*(1._sp- &
			          erf(2._sp*log(sm(1:n_modes1)/smax)/ &
			          (3._sp*sqrt(2._sp)*sig_aer(1:n_modes1)) )))   ! eq 13: of 2000 paper

			act_frac2=1._sp/(n_aer)*(n_aer*5.e-1_sp*(1._sp- &
		     erf(2._sp*log(sm/smax)/(3._sp*sqrt(2._sp)*sig_aer) ))) ! eq 13: of 2000 paper
		     
		else if(method_flag.ge.2) then
			a=4._sp*sigma*molw_vap/(rhow*r_gas*tcb)             ! eq 5: abdul-razzak, ghan
			! above, 2 for abdul-razzak,4 for fountoukis and nenes
  
			b=density_final/ &
			  ( (molw_core*mass_initial/nu_core+ & 
			  sum(molw_org*mass_org_condensed/nu_org)* &
			  	mass_initial/sum(mass_initial(1:n_modes1))) / &
			  	mass_final)/(rhow/molw_vap)                   ! eq 6: abdul-razzak, ghan

			sgi=sqrt(4._sp*a**3/27._sp/(b*d_aer**3))           ! eq 17 fountoukis and nenes

			smax=max(zbrent(fountoukis_nenes,1.e-20_sp,100.e-2_sp,1.e-20_sp),1.e-20_sp)
			!act_frac=brent(10d-2,1.d-10,1.d-20,fountoukis_nenes,1.d-30,smax)
			!smax=max(smax,1.d-20)

			!act_frac=sum(0.5_sp*&
			!   erfc(2_sp*log(sgi/smax)/(3_sp*sqrt(2_sp)*sig_aer))) ! eq 8 and 9 f+n 
			act_frac1=1._sp/sum(n_aer(1:n_modes1))*&
			 sum(n_aer(1:n_modes1)*5.e-1_sp*(1._sp- &
			 erf(2._sp*log(sgi(1:n_modes1)/smax)/ &
			 (3._sp*sqrt(2._sp)*sig_aer(1:n_modes1)) )))   ! eq 13: of 2000 paper

			act_frac2=1._sp/(n_aer)*(n_aer*5.e-1_sp*(1._sp- &
			 erf(2._sp*log(sgi/smax)/(3._sp*sqrt(2._sp)*sig_aer) )))! eq 13: of 2000 paper
		 end if
 
		 act_frac1=act_frac2
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
	end subroutine ctmm_activation
	
	
	
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! diffusivity of water vapour in air										   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the diffusivity of water vapour in air
	!>@param[in] t: temperature
	!>@param[in] p: pressure
	!>@return dd: diffusivity of water vapour in air
	function dd(t,p)
		use nrtype1
		implicit none
		real(sp), intent(in) :: t, p
		real(sp) :: dd
		dd=2.11e-5_sp*(t/273.15_sp)**1.94*(101325._sp/p)
	end function dd

	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! conductivity of water vapour												   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the thermal conductivity of air
	!>@param[in] t: temperature
	!>@return ka: thermal conductivity of air
	function ka(t)
		use nrtype1
		implicit none
		real(sp), intent(in) :: t
		real(sp) :: ka
		ka=(5.69_sp+0.017_sp*(t-273.15_sp))*1.e-3_sp*4.187_sp
	end function ka

	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! dry potential temperature 												   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the thermal conductivity of air
	!> Note uses tinit, pinit, rhinit from module
	!>@param[in] p: pressure (Pa)
	!>@return dry_potential temperature (K)
	function dry_potential(p)
		use nrtype1
		implicit none
		real(sp), intent(in) :: p
		real(sp) :: dry_potential
		real(sp) :: total_water1, total_water2, tcalc 
		total_water1=rhinit*eps*svp(tinit)/(pinit-svp(tinit))

		!print *,'kappa',svp(tinit),eps
		tcalc=tinit*(p/pinit)**kappa

		total_water2=eps*svp(tcalc)/(p-svp(tcalc))

		dry_potential=total_water2-total_water1

	end function dry_potential

	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! saturation vapour pressure over liquid                                       !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the saturation vapour pressure over liquid water according to buck fit
	!>@param[in] t: temperature
	!>@return svp_liq: saturation vapour pressure over liquid water
	function svp(t)
		use nrtype1
		implicit none
		real(sp), intent(in) :: t
		real(sp) :: svp
		svp = 100._sp*6.1121_sp* &
			  exp((18.678_sp - (t-273.15_sp)/ 234.5_sp)* &
			  (t-273.15_sp)/(257.14_sp + (t-273.15_sp)))
	end function svp


	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! 3rd moment - for integration												   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the third moment of a lognormal
	!>@param[in] d: diameter (m)
	!>@return ln3: third moment in a size interval
	function ln3(d)
		use nrtype1
		implicit none
		real(sp), dimension(:), intent(in) :: d
		real(sp), dimension(size(d)) :: ln3

		! add all modes together
		ln3=pi*d**2/(sqrt(twopi*sig_dummy**2)*6.)* &
		exp(-log(d/d_dummy)**2/(2.*sig_dummy**2))* &
		density_dummy*n_dummy
	end function ln3

	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! mass in a lognormal       												   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the mass across one lognormal distribution
	!>@param[in] d1: mode diameter (m)
	!>@return mass_integrate: total mass in the distribution
	function mass_integrate(d1)
		use nrtype1
		use nr1, only : zbrent,qsimp,qromb,midpnt
		implicit none
		real(sp), intent(in) :: d1
		real(sp) :: mass_integrate

		integer(i4b):: i
		d_dummy=d1  ! guess at d_aer, used to calculate the new standard deviation
		sig_dummy=zbrent(find_sig_aer,1.e-9_sp,2._sp,1.e-6_sp) ! 1.d-30
		!  mass_integrate=abs(qromb(ln3,0.d-10,3.d-6)-mass_dummy)
		! moment generating function
		! http://www.mlahanas.de/math/lognormal.htm
		mass_integrate=n_dummy*exp(3._sp*log(d_dummy) + &
					3._sp**2*sig_dummy**2/2._sp) &
				   *density_dummy*pi/6._sp
		mass_integrate=abs(mass_integrate-mass_dummy)

	end function mass_integrate

	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! find sigma for the size distribution										   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the thermal conductivity of air
	!>@param[in] sig_aer_new: geometric standard deviation
	!>@return find_sig_aer: arithmetic standard deviation
	function find_sig_aer(sig_aer_new)
		use nrtype1
		real(sp), intent(in) :: sig_aer_new
		real(sp) :: find_sig_aer
		real(sp) :: sd1
		sd1=exp(log(d_dummy)+0.5_sp*sig_aer_new)*sqrt(exp(sig_aer_new**2)-1._sp)
		find_sig_aer=sd1-sd_dummy
	end function find_sig_aer

	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! fountoukis and nenes integrals											   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the integral in the fountoukis and nenes method
	!>@param[in] smax1: max supersaturation
	!>@return fountoukis_nenes: integral for activation
	function fountoukis_nenes(smax1)
		use nrtype1
		use nr1, only : zbrent,qsimp,qromb,midpnt
		implicit none
		real(sp), intent(in) :: smax1
		real(sp) :: fountoukis_nenes

		real(sp) :: integral, integral1,smax2,spart,upart,umax,i1,i2,discriminant, &
				  deq,dw3
		integer(i4b):: i

		smax2=max(smax1,1e-20_sp)
		alpha_sup=grav*molw_vap*lv/(cp*r_gas*tcb**2)- &
		grav*molw_air/(r_gas*tcb)                ! eq 11 arg
		sigma_sup=r_gas*tcb/(svp(tcb)*molw_vap)+ &
				molw_vap*lv**2/(cp*pcb*molw_air*tcb)  ! eq 11?
		g=rhow*r_gas*tcb/(svp(tcb)*dd(tcb,pcb)*molw_vap) + &
		lv*rhow/(ka(tcb)*tcb)*(lv*molw_vap/(r_gas*tcb)-1_sp)  ! eq 12
		g=4_sp/g
		! do the integral 
		! f-n (2005 use approximate form for integral, here we use quadrature)
		c1(1)=2._sp*a/3._sp
		c1(2)=g/alpha_sup/w
		c1(6)=smax2


		!  discriminant=smax2**4_sp-16._sp*a**2_sp*alpha_sup*w/(9_sp*g)
		discriminant=1._sp-16._sp*a**2._sp*alpha_sup*w/(9._sp*g)/smax2**4._sp

		if(discriminant.ge.0._sp) then
		 spart=smax2*(0.5_sp*(1._sp+ &
		   (1._sp-16_sp*a**2*alpha_sup*w/(9._sp*g*smax2**4 ))**0.5_sp) &
		   )**0.5_sp
		else
		 spart=smax2*min(2.e7_sp*a/3_sp*smax2**(-0.3824_sp),1._sp) 
		endif

		integral=0._sp
		integral1=0._sp
		do i=1,n_mode_s
		  c1(3)=2_sp*n_aer(i)/(3._sp*sqrt(2._sp*pi)*sig_aer(i))
		  c1(4)=sgi(i)
		  c1(5)=2._sp*sig_aer(i)**2
		  ! note for multiple modes, have to calculate the integral below for each mode
		  if(method_flag.eq.3) then
			 integral1=integral1+qromb(integral3_fn,0._sp,smax2)
		  endif

		! approximate method
		  upart=2._sp*log(sgi(i)/spart)/(3._sp*sqrt(2._sp)*sig_aer(i))
		  umax =2._sp*log(sgi(i)/smax2)/(3._sp*sqrt(2._sp)*sig_aer(i))

		  i1=n_aer(i)/2._sp*sqrt(g/alpha_sup/w)*smax2* &
			 (erfc(upart)-5e-1_sp*(sgi(i)/smax2)**2_sp*exp(4.5_sp*sig_aer(i)**2)* &
			 erfc(upart+3_sp*sig_aer(i)/sqrt(2._sp)))
		  i2=a*n_aer(i)/(3_sp*sgi(i))*exp(9_sp/8_sp*sig_aer(i)**2_sp)* &
			 (erf(upart-3_sp*sig_aer(i)/(2._sp*sqrt(2._sp))) - &
			 erf(umax-3._sp*sig_aer(i)/(2._sp*sqrt(2._sp) )) )
		!      print *,'i1, i2',i1,i2, upart, umax

		  if(method_flag.eq.2) then
			 integral1=integral1+(i1+i2)
		  endif   

		  ! for the giant ccn - barahona et al (2010)
		  if(giant_flag.eq.1) then
			dw3  = (sqrt(2._sp)*log(sgi(i)/spart)/3_sp/sig_aer(i))- &
				   (1.5_sp*sig_aer(i)/sqrt(2._sp))
			deq= a*2_sp/sgi(i)/3_sp/sqrt(3._sp)      
			dw3=n_aer(i)*deq*exp(9_sp/8_sp*sig_aer(i)**2._sp)*smax2* &
			   (erfc(dw3))*((g*alpha_sup*w)**0.5_sp)   
		!         
			dw3=dw3/(2._sp*g*smax2)*(g/alpha_sup/w)**0.5_sp
			integral1=integral1 +dw3
		  endif
		enddo
		!  print *,'integral',integral,integral1
		! cost function - eq 10 fountoukis and nenes

		fountoukis_nenes=(2._sp*alpha_sup*w/(pi*sigma_sup*rhow)- &
					   g*smax2*integral1)

	end function fountoukis_nenes

	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! integrand 3 in FN          												   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the integrand in the FN method (for the quadrature case)
	!>@param[in] s: s parameter
	!>@return integral3_fn: integrand in FN (for use with quadrature)
	function integral3_fn(s)
		use nrtype1
		implicit none
		real(sp), dimension(:), intent(in) :: s
		real(sp), dimension(size(s)) :: integral3_fn

		integral3_fn=((c1(1)/(s+1.e-50_sp))**2+c1(2)*(c1(6)**2-s**2))**0.5_sp* &
				   c1(3)/(s+1.e-50_sp)*exp(-log((c1(4)/(s+1.e-50_sp))**(2._sp/3._sp))**2/c1(5)) ! eq 14 f-n
	end function integral3_fn
		
		
	
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculate the condensed organics (in Crooks, GMD 2016)
	!>parameterisation developed at university of manchester
	!>@param[in] n_modes1: number of aerosol modes
	!>@param[in] n_sv1: number of volatility bins
	!>@param[in] org_content1: amount of organic in volatility bins
	!>@param[in] nu_org1: van hoff factor in volatility bins
	!>@param[in] molw_org1: molecular weight in volatility bins
	!>@param[in] mass_core: mass in aerosol modes
	!>@param[in] nu_core1: van hoff factor in core
	!>@param[in] molw_core1: molecular weight of modes (core only)
	!>@param[in] s1, t1: rh and temperature.
	!>@param[inout] mass_org_condensed1: condensed organics
	subroutine solve_semivolatiles(n_modes1,n_sv1, &
					org_content1, log_c_star1, delta_h_vap1, &
					nu_org1, molw_org1, &
					mass_core1, nu_core1, molw_core1,s1, t1, &
					mass_org_condensed1)
		use nrtype1
		use nr1, only : zbrent, brent
		implicit none
		integer(i4b), intent(in) :: n_modes1, n_sv1
		real(sp), dimension(n_sv1), intent(in) :: org_content1, nu_org1, molw_org1, &
							log_c_star1, delta_h_vap1
		real(sp), dimension(n_modes1), intent(in) :: mass_core1, nu_core1, molw_core1
		real(sp), intent(in) :: s1, t1
		real(sp), dimension(n_sv1), intent(inout) :: mass_org_condensed1
		
		
		real(sp) :: ct
		real(sp), dimension(n_sv1) :: c_c
		
		
		! set variables in module (for passing to optimizer)
		nu_org=nu_org1
		molw_org=molw_org1
		nu_core=nu_core1
		molw_core=molw_core1
		s=s1
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		
		cstar = 10._sp**log_c_star1* (298.15_sp/t1) * &
					exp(-delta_h_vap1*1.e3_sp/r_gas *(1._sp/t1-1._sp/298.15_sp))
										 ! c* needs to be adjusted by delta_h_vap / t
		c_ions=org_content1*nu_org1/molw_org1 ! c - all ions
		c0=sum(mass_core1*nu_core1/molw_core1)  ! number of "core" ions
		ct=1._sp/(1._sp-s)*(sum(c_ions)+c0)  ! equation 5 from Crooks et al. (2016, GMD)
										! basically saturation ratio is mole fraction
		! ct is the total concentration of all ions in the condensed phase
		! solve iteratively to find CT in matt's paper
		ct=abs(zbrent(partition01,ct,1._sp/(1._sp-s)*c0,1.e-8_sp))
		!xmin=brent(1.e-15_sp,1.e-8_sp,ct*5._sp, &
		!					partition01,1.e-10_sp,ct)
		
		epsilon1=(1._sp+cstar/ct)**(-1) ! partitioning coefficients
		c_c=c_ions*epsilon1   ! condensed

		mass_org_condensed1=c_c/nu_org1*molw_org1
	end subroutine solve_semivolatiles
	
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	! partition01          										        		   !
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>calculates the partitioning coefficient according to Crooks et al (2016 GMD)
	!>@param[in] ct: total ions, including water
	!>@return guess, minus calculated - for root-finder
	function partition01(ct)
		use nrtype1
		implicit none
		real(sp), intent(in) :: ct
		real(sp), dimension(size(epsilon1)) :: c_c
		real(sp) :: partition01
		
		real(sp) :: ct1, ct2
		ct2=abs(ct)
		epsilon1 = (1._sp+cstar/(ct2))**(-1)
		c_c=(c_ions)*epsilon1
		ct1=1._sp/(1._sp-s)*(sum(c_c*nu_org)+c0)
		
		partition01=(ct1-ct2)
	
	end function partition01
	
	
		
		
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>allocate arrays for activation code
	!>@param[in] n_modes: number of aerosol modes
	!>@param[in] n_sv: number of organic / volatility modes
	!>@param[inout] n_aer1: number in modes
	!>@param[inout] d_aer1: diameter in modes
	!>@param[inout] sig_aer1: geo std in modes
	!>@param[inout] molw_core1:molw in core
	!>@param[inout] density_core1: solute density
	!>@param[inout] nu_core1: van hoff factor
	!>@param[inout] org_content1: organic content in vol bins
	!>@param[inout] molw_org1: molw in volatility bins
	!>@param[inout] density_org1: density in volatility bins
	!>@param[inout] delta_h_vap1: enthalpy in volatility bins
	!>@param[inout] nu_org1: van hoff factor in volatility bins
	!>@param[inout] log_c_star1: log_c_star in volatility bins
	!>@param[inout] act_frac1: activated fraction in modes
	subroutine allocate_arrays(n_mode,n_sv,n_aer1,d_aer1,sig_aer1, &
			molw_core1,density_core1,nu_core1,org_content1, &
			molw_org1, density_org1,delta_h_vap1,nu_org1,log_c_star1, act_frac1)
		use nrtype1
		implicit none
		integer(i4b), intent(in) :: n_mode, n_sv
		real(sp), dimension(:), allocatable, intent(inout) :: n_aer1,d_aer1,sig_aer1, &
							molw_core1, density_core1, nu_core1, org_content1, &
							molw_org1, density_org1, delta_h_vap1, nu_org1, log_c_star1, &
							act_frac1
		
		integer(i4b) :: AllocateStatus
		allocate( n_aer(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( d_aer(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( sig_aer(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( n_aer1(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( d_aer1(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( sig_aer1(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( d_aer_new(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( sgi(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( density_final(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( mass_initial(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( mass_final(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( sd(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( b(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( sm(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( eta(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( f1(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( f2(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( density_core(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( density_core1(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( molw_core(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( molw_core1(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( nu_core(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( nu_core1(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( act_frac(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( act_frac1(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( act_frac2(1:n_mode), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		
		allocate( molw_org(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( molw_org1(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( r_org(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( log_c_star(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( log_c_star1(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( cstar(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( c_ions(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( epsilon1(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( org_content(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( org_content1(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( density_org(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( density_org1(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( nu_org(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( nu_org1(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( mass_org_condensed(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( delta_h_vap(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		allocate( delta_h_vap1(1:n_sv), STAT = AllocateStatus)
		if (AllocateStatus /= 0) STOP "*** Not enough memory ***"	
		
	
	end subroutine allocate_arrays
	
	!>@author
	!>Paul J. Connolly, The University of Manchester
	!>@brief
	!>initialise arrays for activation code
	!>@param[in] n_modes: number of aerosol modes
	!>@param[in] n_sv: number of volatility bins
	!>@param[in] p1: pressure (Pa)
	!>@param[in] t1: temperature (K)
	!>@param[in] w1: vertical wind (m/s)
	!>@param[in] n_aer1: number concentration in modes
	!>@param[in] d_aer1: diameter in modes
	!>@param[in] sig_aer1: geometric standard deviation in modes
	!>@param[in] molw_org1: molecular weight in volatility bins
	!>@param[in] density_core1: density in modes
	subroutine initialise_arrays(n_modes,n_sv,p1,t1,w1,n_aer1, &
								d_aer1,sig_aer1, molw_org1,density_core1)
		use nrtype1
		implicit none
		integer(i4b), intent(in) :: n_modes, n_sv
		real(sp), intent(in) :: p1,t1,w1
		real(sp), dimension(n_modes), intent(in) :: n_aer1,d_aer1,sig_aer1, density_core1
		real(sp), dimension(n_sv), intent(in) :: molw_org1
		
		integer(i4b) :: i
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		! define initial conditions                                                      !
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		rhinit=0.999_sp                                ! as a fraction - assume we are
													   ! assume we are at cb.
		pinit=p1                                       ! pascals
		tinit=t1                                       ! kelvin
		w    =w1                                       ! m s-1
		n_aer=n_aer1
		d_aer=d_aer1
		sig_aer=sig_aer1
		molw_org=molw_org1
		density_core=density_core1
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		
		
		
		
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		! define the organic properties                                                  !
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		!log_c_star=(/(i,i=-0,3)/)                      ! watch out for type?
		!nu_org=1._sp                                    ! disociation factor
		!molw_org=200e-3_sp                                ! kg per mol
		r_org=r_gas/molw_org
		!density_org=1500._sp                            ! kg m-3
		!delta_h_vap=150._sp                             ! enthalpy phase change (kj mol-1)
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		
		
		

		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		! find the t & p at cloud base                                                   !
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		pcb=pinit !zbrent(dry_potential,10000._sp,pinit,1.d-8)
		tcb=tinit !tinit*(pcb/pinit)**kappa
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		! initial mass in ith distribution                                               !
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		do i=1,n_modes
			density_dummy=density_core(i)                  ! the density of the core dist.
			n_dummy=n_aer(i)
			sd_dummy=sig_aer(i)
			d_dummy=d_aer(i)
			! initial mass in ith distribution
			! moment generating function
			! http://www.mlahanas.de/math/lognormal.htm
			mass_initial(i)=n_dummy*exp(3._sp*log(d_dummy) + &
							3._sp**2_sp*sd_dummy**2/2._sp) &
						   *density_dummy*pi/6._sp
		enddo
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


	end subroutine initialise_arrays
	
	
	end module sub	

