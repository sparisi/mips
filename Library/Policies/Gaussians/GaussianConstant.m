classdef (Abstract) GaussianConstant < Gaussian
% GAUSSIANCONSTANT Generic class for Gaussian policies. Its parameters do 
% not depend on any features, i.e., Action = N(mu, Sigma).
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        mu
        Sigma
        U     % Cholesky decomposition
    end
    
    methods
        
        %% LOG(PDF)
        function logprob = logpdf(obj, Actions)
            Actions = bsxfun(@minus, Actions, obj.mu);
            d = size(Actions,1);
            Q = obj.U' \ Actions;
            q = dot(Q,Q,1); % quadratic term (M distance)
            c = d * log(2*pi) + 2 * sum(log(diag(obj.U))); % normalization constant
            logprob = -(c+q)/2;
        end

        %% MVNPDF
        function probability = evaluate(obj, Actions)
            probability = exp(obj.logpdf(Actions));
        end

        %% MVNRND
        function Actions = drawAction(obj, N)
            Actions = bsxfun(@plus, obj.U'*randn(obj.daction,N), obj.mu);
        end
        
        %% PLOT PDF
        function fig = plot(obj)
        % Plot N(mu(i),Sigma(i,i)) for each dimension i (NORMPDF)
            fig = figure(); hold all
            xlabel 'x'
            ylabel 'pdf(x)'
            range = ndlinspace(obj.mu - 3*diag(obj.Sigma), obj.mu + 3*diag(obj.Sigma), 100);
            
            norm = bsxfun(@times, exp(-0.5 * bsxfun(@times, ...
                bsxfun(@minus,range,obj.mu), 1./diag(obj.Sigma)).^2), ... 
                1./(sqrt(2*pi) .* diag(obj.Sigma)));
            plot(range', norm')
            legend show
        end

        %% KL
        function d = kl(p, q, varargin)
        % Kullback–Leibler divergence divergence KL(P||Q) from distribution 
        % P to distribution Q
        %
        % http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
            assert(ismember('GaussianConstant',superclasses(q)) == 1, 'Both distributions must be Gaussian.')
            k = p.daction;
            assert(k == q.daction, 'The Gaussians must have the same dimension.')
            s0 = p.Sigma;
            s1 = q.Sigma;
            m0 = p.mu;
            m1 = q.mu;
            d = 0.5 * (trace(s1 \ s0) + (m1 - m0)' / s1 * (m1 - m0) - k + log(det(s1) / det(s0)));
        end
        
    end

end
