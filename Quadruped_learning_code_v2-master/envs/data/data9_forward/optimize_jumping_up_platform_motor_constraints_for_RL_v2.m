% for generating data for reinforcement learning experiments
% different from RL version, it adds:
% add dynamic constraints from https://arxiv.org/pdf/2104.04644.pdf, which
% consider power consumption direction

% our work also consider power supply limits (power, and voltage limits)

% for generating data for reinforcement learning experiments

clc; close all;
% clear all; use this if change dyanmics
robot = 2; %(0=Cheetah 3; 1= AlienGo; 2=A1)
option.platform_height=0.0;
option.jumping_length= 0.6; %0.6
option.pitch_end_offset= 0; % clockwise is positive
% option.jumping_length= -0.4; option.platform_height= 0;  % backflip
% option.pitch_end_offset=-2*pi-pi/6; % clockwise is positive

option.jump_with_4_legs=0;
option.knee_sign=1; % 1: knee backward, -1: knee forward

% integration settings
res = 1; % change to increase/decrease timestep length without changing timing
dt = 0.01/ res; % time between timesteps

option.use_voltage_constraint= 1;
option.use_current_constraint= 0;
option.use_upper_power_constraint=1;
option.use_lower_power_constraint=0;

option.voltage_max =21.5;
option.current_max =60;

% motor configuration
% V = I*R+L*dI/dt+Ke*w, where Kt=T/I, Ke=Kt
% Thus w=V/Kt-T*R/Kt^2 - L*dT/(Kt^2*dt)
% Approximation: w=V/Kt-T*R/Kt^2

Kt=4/34; % from Unitree plot
torque_motor_max = 4;
speed_motor_max = 1700*2*pi/60; % max speed motor --> rad/s
gear_ratio =8.5; % joint_speed=speed_motr/gear_ratio, by 1740*2*pi/60/21: from Unitree plot

%% torque and joint speed constraints
% eq: joint_speed=t2j_motor*x+b ; % rad/s
max_js= speed_motor_max; 
min_js=940*2*pi/60; % to compute the slope of line
alpha_motor= (min_js-max_js)/(torque_motor_max-0.2); % slope of the line
bm=(min_js-alpha_motor*torque_motor_max); % where bm=V/Kt=21.4/Kt=182 (rad/s)


double_contact = 60*res;
single_contact = 30*res;
flight_phase = 40*res;

option.file_name='jumpingFull_A1_10ms_h00_d60';


if robot == 1 % AlienGo
    l2=0.25; l3=0.25; % leg lengths
%     option.q_joint_limit = [3.14; 2.5; 3.14; 2.5];
    option.q_joint_limit_lower = [-3.14; 0.64578; -3.14; 0.64578];
    option.q_joint_limit_upper = [3.14; 2.775; 3.14; 2.775];
    option.torque_saturation_stance= 1*[44;55;44;55];
    option.torque_saturation_swing= 20; %8;%8.5;
    option.torque_saturation_flight= 20; %8;%21;
    option.joint_vel_limit = 16;

elseif robot == 2 % A1 robot
    l2=0.2; l3=0.2; % leg lengths
    %option.q_joint_limit_lower = -[3.14; 2.5; 3.14; 2.5];
    %option.q_joint_limit_upper = [3.14; 2.5; 3.14; 2.5];
    option.q_joint_limit_lower = [-4.1888; 0.916; -4.1888; 0.916];
    option.q_joint_limit_upper = [1.047; 2.697; 1.047; 2.697];
    option.torque_saturation_stance= 1*[33.5;33.5;33.5;33.5];
    option.torque_saturation_swing= 10; %8;%8.5;
    option.torque_saturation_flight= 15; %8;%21;
    option.joint_vel_limit = 21;
    
else % Cheetah 3
    l2=0.342; l3=0.345; % leg lengths
    option.q_joint_limit_lower = -[3.14; 2.5; 3.14; 2.5];
    option.q_joint_limit_upper = [3.14; 2.5; 3.14; 2.5];
    option.torque_saturation_stance= 1*[187;216;187;216];
    % option.torque_saturation_stance= 170*ones(4,1);
    option.torque_saturation_swing= 8.5;
    option.torque_saturation_flight= 21;

    option.joint_vel_limit = 18;
end
% option.q_end=[-pi/4;pi/2;-pi/4;pi/2];%[-pi/6;pi/3;-pi/6;pi/3];
% option.q_end=[-pi/3;2*pi/3;-pi/3;2*pi/3];
option.q_init=.5*[0 0 0 -pi*.8 1.5*pi -pi*.8 1.5*pi]'; 
option.q_init=option.q_init*option.knee_sign;
option.q_ref=option.q_init(4:7); % reference q for cost function

platform_height=option.platform_height;

% system parameter options
option.joint_passive_damping= 1;


N=double_contact+single_contact+flight_phase;

option.q_ref=option.q_ref+0*[0.8;-0.5;0.8;-0.5];

option.q_end=option.q_init(4:7);
option.q_end=option.q_end+0*[0.8;-0.5;0.8;-0.5];

option.posing_time = 10;
option.mu=0.6;
option.use_torque_saturation_constraint=1;
% real hardware torque saturation: tau_motor_size_max=24.5
% tau_hip_max=24.5*7.6666=187.8
% tau_knee_max=24.5*8.846=216.7

option.res=res;
option.N=N;
option.double_contact=double_contact;
option.single_contact=single_contact;

option.q_init=option.q_init*option.knee_sign;
option.q_end=option.q_end*option.knee_sign;
% option.landing_height_offset= (l2+l3)*cos(option.q_end(1))-0.2623; %final height offset(difference between landing and initial COM height)
h_com_init=l2*cos(option.q_init(4))+l3*cos(option.q_init(4)+option.q_init(5));
h_com_end=l2*cos(option.q_end(1))+l3*cos(option.q_end(1)+option.q_end(2));
option.landing_height_offset= h_com_end - h_com_init; %final height offset(difference between landing and initial COM height)
 
%%initialize_optimization;
% run 'clear' before running this script if you've changed the dynamics

if computer == 'PCWIN64'
     addpath(genpath('casadi-windows-matlabR2016a-v3.5.1')); % for Windows
else
    addpath(genpath('casadi')); % for Linux
end    

% add spatial v2 library (should work on all OS)
addpath(genpath('spatial_v2'));

%% Symbolic Dynamics
% only do symbolic dynamics if we need to
if ~(exist('symbolic_done','var'))
    disp_box('Generate Symbolic Dynamics');
    tic;
    params = get_robot_params(robot);  % robot masses, lengths... 
    model  = get_robot_model(params); % spatial_v2 rigid body tree + feet
    [H,C,p,pfs,Jf,Jdqdf,vfs] = get_simplified_dynamics(model); % symbolic functions for dynamics
    addpath(genpath('dynamics_out'));
    write_dynamics_to_file(H,C,p,pfs,Jf,Jdqdf,vfs);
    toc;
    
    symbolic_done = 1;
end

%% Contact schedule

flight = N-double_contact-single_contact; % timesteps spent in the air
% build contact schedule (it needs to be size N+1 because of constraints)
cs = [2*ones(1,double_contact) ones(1,single_contact) zeros(1,flight) 0];


%% Initial conditions and integration settings

% zero spatial force on both feet (used as input to dynamics)
zero_force = {zeros(6,1),zeros(6,1)};


% set initial joint position
q_init = option.q_init; %.5*[0 0 0 -pi*.8 1.5*pi -pi*.8 1.5*pi]'; 
% find initial foot location using forward kinematics
[~,~,~,~,~,pfi,~,~,~] = all_the_dynamics(model,q_init,zeros(7,1),zero_force,0);
% only care about x,z position of feet.
pfi{1} = pfi{1}([1 3]);
pfi{2} = pfi{2}([1 3]);

%% Optimization Variables

% create optimization object
opti = casadi.Opti();
% create optimization variables
X = opti.variable(7*3 + 4 + 4,N);

% names for optimization variables
ddq = X(1:7,:);    % joint acceleration (includes floating base coordinates)
dq  = X(8:14,:);   % joint velocity
q   = X(15:21,:);  % joint position
f_f = X(22:23,:);  % front foot force
f_r = X(24:25,:);  % rear foot force
tau_joint = X(26:29,:); % actuator torques


% inputs to the symbolic dynamics
Q = [q;dq]; % Q is q, q_dot for all timesteps
F_0 = zeros(6,N);  % zero external force
C_in = [Q;F_0];    % input to the function that calculates C, external forces are in J'*f, not in C for this formulation!

% friction of ground
mu = option.mu;

% Constraints
disp_box('Building constraints');
tic;

% damping of joints
Kd_j = diag([0 0 0 1 1 1 1])*option.joint_passive_damping;

%% Cost Function
% desired end position
% q_end_joints = .5*[0 0 0 -pi/6 pi/3 -pi/6 pi/3]'; 
q_end_joints = q_init; 
q_end_des = [q_end_joints(4:7)];

% error in joints at last timestep (not including body coordinates)
qerr = q(4:7,N) - option.q_end;%q_end_des;
% minimize error^2
J=100*qerr'*qerr;


option.alpha_tau=0.05; % weight in cost function for tau

for k=1:N-1
    qerr = q(4:7,k) - option.q_ref;
    tau_k=tau_joint(:,k);
    J=J+1*qerr'*qerr+option.alpha_tau*tau_k'*tau_k;
%     J=J+1*qerr'*qerr+0.0001*tau_k'*tau_k;
end

opti.minimize(J);

%% set_constraints;
% Loop through timesteps, applying constraints
for k = 1:N-1
    
    disp([num2str(k) ' of ' num2str(N-1) ' cs: ' num2str(cs(k))]);
    
    % the 'k' suffix indicates the value of the variable at the current
    % timestep
    qk = q(:,k); % q at timestep k
    Qk = Q(:,k); % q,dq at timestep k
    
    qjk = Qk(4:7); % q for the joints at timestep k
    dqjk = Qk(11:14); % dq for the joints at timestep k
    
    pf1k = pf_sym1(qk); % position of front foot
    pf2k = pf_sym2(qk); % position of rear foot
    Hek = H_sym(qk);    % mass matrix
    
    p_knee1_k=p_sym5(qk); % position of front knee (Need to check with Jared!!!)
    p_knee2_k=p_sym7(qk); % position of rear knee
    
    p_hip1_k=p_sym4(qk); % front hip position
    p_hip2_k=p_sym6(qk); % rear hip position
    
    % modify mass matrix to include rotors
    Hek = add_offboard_rotors(Hek,2*params.I_rot,8,[0 0 0 1 1 1 1]); 
    % bias torques
    Cek = C_sym(C_in(:,k));
    % jacobians
    J1ek = Jf_sym1(qk);
    J2ek = Jf_sym2(qk);
    % stack jacobians, remove y coordinate
    Jfek = [J1ek([1 3],:); J2ek([1 3],:)];
    % damping torque
    tau_d = 1*Kd_j * Qk(8:end);
    
    % build Aek, xek such that Aek*xek = bek:
    % notice that we use cs(k+1) here because the integration puts in a 
    % 1 timestep delay
    % if we're in flight...
    if(cs(k+1) == 0)
        % constraints are just ddq, H
        Aek = Hek;
        xek = ddq(:,k);
        % ground reaction forces must be zero
        opti.subject_to(f_f(:,k) == [0;0]);
        opti.subject_to(f_r(:,k) == [0;0]);
    % if we have back feet on the ground
    elseif(cs(k+1) == 1)
        % front feet reaction forces must be zero
        opti.subject_to(f_f(:,k) == [0;0]);
        % constraints now include rear reaction force and jacobian
        Aek = [Hek, J2ek([1 3],:).'];
        xek = [ddq(:,k);f_r(:,k)];
        % rear feet reaction forces must be at least 30 N to prevent slip
        opti.subject_to(f_r(2,k) <= -50);
    % if both feet on the ground
    else
        % constraints include both reaction forces and jacobian
        Aek = [Hek, Jfek.'];
        opti.subject_to(f_f(2,k) <= -30);
        opti.subject_to(f_r(2,k) <= -50);
        xek = [ddq(:,k);f_f(:,k);f_r(:,k)];
    end
    
    % bek is always torque (bias + damping + motor)
    % the body coordinates aren't actuated
    bek = [-Cek - tau_d + [0;0;0;tau_joint(:,k)]];
    
    % euler integration of dq,q
    dq_int = dq(:,k) + dt * ddq(:,k);
    q_int  = q(:,k)  + dt * dq_int;
    
    % integrated Q
    Q_int = [q_int;dq_int];
    
    % integration constraint
    Q_next = [q(:,k+1);dq(:,k+1)];
    opti.subject_to(Q_next == Q_int);
    
    % dynamics constraint
    opti.subject_to(Aek * xek == bek);
    
    % constraint feet to their initial position, if appropriate
    if(cs(k) >= 1)
        opti.subject_to((pf2k([1 3]) - pfi{2}) == [0;0]);
    end
    
    if(cs(k) >= 2)
        opti.subject_to((pf1k([1 3]) - pfi{1}) == [0;0]);
    end

    % constraint on knee positions to be always above the ground
    if (cs(k) ~= 0) % if not the flight phase
        if robot == 2 % A1 robot
           opti.subject_to(p_knee1_k(3)-pfi{1}(2)>=0.05);
           opti.subject_to(p_knee2_k(3)-pfi{2}(2)>=0.05);
           
           opti.subject_to(p_hip1_k(3)-pfi{1}(2)>=0.07);
           opti.subject_to(p_hip2_k(3)-pfi{2}(2)>=0.07);
        else
           opti.subject_to(p_knee1_k(3)-pfi{1}(2)>=0.1);
           opti.subject_to(p_knee2_k(3)-pfi{2}(2)>=0.1);
        end

    end

    % constraint on rear swing foot clearance 
    % in the first several iterations on the flight phase
    flight_phase_index=double_contact+single_contact+1;
    if robot == 0
        if (k>=flight_phase_index) && (k<=flight_phase_index+7)
           opti.subject_to(pf2k(3)-pfi{2}(2)>=0.02*(k-flight_phase_index+1));
        end        
    end
    
    % max torque
    % both feet on the ground, full torque all motors
    if option.use_torque_saturation_constraint
        if(cs(k) == 2)
%             opti.subject_to(tau_joint(:,k) <= 2*option.torque_saturation_stance); 
%             opti.subject_to(tau_joint(:,k) >= -2*option.torque_saturation_stance);
            
            opti.subject_to(tau_joint(:,k) <= ...
                2*[25; 
                   25;
                   option.torque_saturation_stance(3);
                   option.torque_saturation_stance(4)]); 
            opti.subject_to(tau_joint(:,k) >=...
               -2*[25; 
                   25;
                   option.torque_saturation_stance(3);
                   option.torque_saturation_stance(4)]);

        % rear feet on the ground, less torque on swing legs
        elseif(cs(k) == 1)
            opti.subject_to(tau_joint(:,k) <= ...
                2*[option.torque_saturation_swing; 
                   option.torque_saturation_swing;
                   option.torque_saturation_stance(3);
                   option.torque_saturation_stance(4)]); 
            opti.subject_to(tau_joint(:,k) >=...
               -2*[option.torque_saturation_swing; 
                   option.torque_saturation_swing;
                   option.torque_saturation_stance(3);
                   option.torque_saturation_stance(4)]); 
            % needed to make the robot actually jump up.
    %         opti.subject_to(Qk(2) >= 0.1);
        else
            % in flight, low torque limits
            opti.subject_to(tau_joint(:,k) <= 2*[1;1;1;1]*option.torque_saturation_flight); %7*[6;6;6;6]);
            opti.subject_to(tau_joint(:,k) >= -2*[1;1;1;1]*option.torque_saturation_flight); %-7*[6;6;6;6]);
        end
    end
   
    if option.use_lower_power_constraint
            power=0;
            for i=1:4
                power = power+ 0.5*tau_joint(i,k)*dqjk(i)+ (12.5/gear_ratio^2)*(0.5*tau_joint(i,k))^2;
            end
            opti.subject_to(0 <= 2*power); % no charge back to the battery + power limits

    end

    if option.use_upper_power_constraint
            power=0;
            for i=1:4
                power = power+ 0.5*tau_joint(i,k)*dqjk(i)+ (12.5/gear_ratio^2)*(0.5*tau_joint(i,k))^2;
            end
            opti.subject_to(2*power <= option.voltage_max*option.current_max); % no charge back to the battery + power limits

    end

    if option.use_voltage_constraint
            voltage = (0.5*tau_joint(:,k)*alpha_motor/gear_ratio +dqjk*gear_ratio)*Kt;
            opti.subject_to(-[1;1;1;1]*option.voltage_max <= voltage<=[1;1;1;1]*option.voltage_max);
            %opti.subject_to(dqjk <= [1;1;1;1]*0.8*option.joint_vel_limit);
            %opti.subject_to(dqjk >=
            %-[1;1;1;1]*0.8*option.joint_vel_limit);
    end

    % joint velocity
    opti.subject_to(dqjk <= [1;1;1;1]*option.joint_vel_limit);
    opti.subject_to(dqjk >= -[1;1;1;1]*option.joint_vel_limit);

    
%     if option.use_current_constraint
%         opti.subject_to(-option.current_max<= ones(1,4)*tau_joint(:,k)/(gear_ratio*Kt) <=option.current_max);
%     end
    
    % friction cone
    
    opti.subject_to(f_f(1,k) <= - mu*f_f(2,k));
    opti.subject_to(f_f(1,k) >= mu*f_f(2,k));
    
    opti.subject_to(f_r(1,k) <= - mu*f_r(2,k));
    opti.subject_to(f_r(1,k) >= mu*f_r(2,k));   
    
    
    % joint angle limit    
%   Soft Limit from TI Board control:  q_min[3] = {-.5f, -3.14f, -2.5f}; q_max[3] = {.5f, 3.14f, 2.5f};
    qj_min = option.q_joint_limit_lower; %[-3.14; -2.5; -3.14; -2.5];
    qj_max = option.q_joint_limit_upper; %[3.14; 2.5; 3.14; 2.5];
    opti.subject_to(qjk <= qj_max);
    opti.subject_to(qjk >= qj_min);

    % constraint pitch angle
%     opti.subject_to(qk(3) >=-1.2); %? degree
end


% Initial/terminal constraints
opti.subject_to(q(:,1) == q_init);    % inital configuration
opti.subject_to(dq(:,1) == zeros(7,1)); % initial velocity
% opti.subject_to(q(3,N) == -2*pi); % flipped at the end
opti.subject_to(q(3,N) == option.pitch_end_offset); % jumping, dont flipped

%% end position
opti.subject_to(q(2,N) == platform_height+option.landing_height_offset); % end position  
opti.subject_to(q(1,N) >= option.jumping_length);  % end position

%       opti.subject_to(q(4:7,N) == q_init(4:7));  % this is now in the cost
opti.subject_to(q(4:7,(N-option.posing_time):N) == option.q_end);
opti.subject_to(dq(4:7,(N-option.posing_time):N) == zeros(4,1));

toc;

%% Initial guess
opti.set_initial(q,repmat(q_init,1,N));
opti.set_initial(tau_joint,repmat(2*ones(4,1),1,N));


%% Solve!
disp_box('Starting IPOPT');
p_opts = struct('expand',true);
s_opts = struct('max_iter',1500);
opti.solver('ipopt',p_opts,s_opts);
tic;
sol = opti.solve();
toc;

Xs = sol.value(X);
Qs = sol.value(Q);
Ffs = sol.value(f_f);
Frs = sol.value(f_r);
taus = sol.value(tau_joint);
Fs = [Ffs;Frs];

% Compute optimized pf and vf
% NOTE: Need to change the sign of joint position to match with the ROS
pfs=zeros(4,N);
vfs=zeros(4,N);
vf1=zeros(3,N); 
vf3=zeros(3,N);
for i=1:N
    % get desired (optimized) joint position
    qf=[0;-Qs(4:5,i)]; % two front leg
    qr=[0;-Qs(6:7,i)]; % two rear leg
    
    % get desired (optimized) joint velocity
    dq_f=[0;-Qs(11:12,i)]; % two front leg
    dq_r=[0;-Qs(13:14,i)]; % two rear leg
    
    % get foot position
    [Jf1,pf1]=computeLegJacobianAndPosition(qf,1);
    [Jf3,pf3]=computeLegJacobianAndPosition(qr,3);
    pfs(:,i)=[pf1(1),pf1(3), pf3(1),pf3(3)]';
    
    % get foot velocity
    vf1(1:3,i)=Jf1*dq_f;
    vf3(1:3,i)=Jf3*dq_r;
    
    vfs(:,i)=[vf1(1,i),vf1(3,i),vf3(1,i),vf3(3,i)]';
end


save(option.file_name,'Xs','Qs','Ffs','Frs','taus','Fs','pfs','vfs','pfi','model','zero_force','option');

disp_box('Animate the result!');

animate_jumping_up_platform;

% max_com_height=max(abs(Qs(2,:)))+0.2623
make_plots;

% write_results_to_file_RL;

% write_results_to_file;
% write_results_to_file_ROS_learning;
