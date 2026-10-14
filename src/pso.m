clc;
clear;
close all;

% 1. Define Discrete Power-of-Two Hardware Grid
hardware_grid = [8, 16, 32, 64, 128, 256];

% 2. Search Space Setup for 3 FFNN Hidden Layers [N1, N2, N3]
nVar = 3;
VarSize = [1 nVar];
VarMin = [8, 8, 8];       % Minimum neuron lower bound
VarMax = [256, 256, 256]; % Maximum neuron upper bound

% Cost Function Handle: Call PINN test_beam (without plotting during search)
CostFunction = @(x) test_beam(x, false);

%% PSO Search Parameters
MaxIt = 10;    % Max iterations
nPop = 15;     % Population size (balanced for neural network training time)

% PSO Coefficients
w = 0.8;        % Inertia weight
wdamp = 0.98;   % Inertia damping factor
c1 = 1.5;       % Personal cognitive factor
c2 = 2.0;       % Social learning factor

% Velocity limits
VelMax = 0.1 * (VarMax - VarMin);
VelMin = -VelMax;

%% Initialisation
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];

particle = repmat(empty_particle, nPop, 1);
GlobalBest.Cost = inf;

disp('Initializing Discrete Hardware-Aware Particle Swarm...');
for i = 1:nPop
    % Generate continuous initial positions within bounds
    raw_pos = VarMin + rand(VarSize) .* (VarMax - VarMin);
    
    % Map directly to nearest discrete power-of-two hardware grid
    particle(i).Position = snap_to_grid(raw_pos, hardware_grid);
    particle(i).Velocity = zeros(VarSize);
    
    % Evaluate initial PINN cost
    particle(i).Cost = CostFunction(particle(i).Position);
    
    % Set Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    
    % Track Global Best
    if particle(i).Best.Cost < GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
end

BestCost = zeros(MaxIt, 1);

%% Main Discrete PSO Search Loop
disp('Starting Optimization Loop...');
for it = 1:MaxIt
    for i = 1:nPop
        % Update particle velocity
        particle(i).Velocity = w * particle(i).Velocity ...
            + c1 * rand(VarSize) .* (particle(i).Best.Position - particle(i).Position) ...
            + c2 * rand(VarSize) .* (GlobalBest.Position - particle(i).Position);
        
        % Velocity clamping
        particle(i).Velocity = max(particle(i).Velocity, VelMin);
        particle(i).Velocity = min(particle(i).Velocity, VelMax);
        
        % Update continuous position
        raw_pos = particle(i).Position + particle(i).Velocity;
        raw_pos = max(raw_pos, VarMin);
        raw_pos = min(raw_pos, VarMax);
        
        % Enforce discrete hardware mapping (power of 2)
        particle(i).Position = snap_to_grid(raw_pos, hardware_grid);
        
        % Evaluate architecture cost via test_beam
        particle(i).Cost = CostFunction(particle(i).Position);
        
        % Personal best update
        if particle(i).Cost < particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
            
            % Global best update
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
    end
    
    % Record iteration cost
    BestCost(it) = GlobalBest.Cost;
    
    % Display iteration statistics
    disp(['Iteration ' num2str(it) ': Best PINN Cost = ' num2str(BestCost(it)) ...
          ' | Optimal Architecture = [' num2str(GlobalBest.Position) ']']);
    
    % Decay inertia factor
    w = w * wdamp;
end

%% Final Results & Deliverable Visualization
disp('--- Optimization Complete ---');
disp(['Optimal Neural Architecture: [N1, N2, N3] = [' num2str(GlobalBest.Position) ']']);
disp(['Final Fitness Cost: ' num2str(GlobalBest.Cost)]);

% Save optimal architecture parameters
BestSol.Position = GlobalBest.Position;
BestSol.Cost = GlobalBest.Cost;
save('optimized_ffnn_pinn.mat', 'BestSol');

% 1. Plot PSO Convergence curve
figure('Name', 'PSO Convergence Curve');
semilogy(BestCost, 'b-o', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best PINN Cost'); grid on;
title('PINN Discrete Hardware-Aware PSO Convergence');

% 2. Evaluate and plot final PINN radiation pattern comparison
disp('Generating final beamforming radiation pattern comparison plots...');
test_beam(GlobalBest.Position, true);

%% Helper Function for Hardware Grid Snapping
function snapped = snap_to_grid(pos, grid)
    snapped = zeros(size(pos));
    for k = 1:numel(pos)
        [~, idx] = min(abs(grid - pos(k)));
        snapped(k) = grid(idx);
    end
end