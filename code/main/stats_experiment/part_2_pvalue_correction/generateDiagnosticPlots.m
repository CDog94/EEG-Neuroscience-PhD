function generateDiagnosticPlots(cfg)
% GENERATEDIAGNOSTICPLOTS
%
% GPU–accelerated Monte–Carlo diagnostic visualisation.
%
% Column layout per distribution (9 rows × 4 cols):
%
%   Col 1 — 10 PDFs from ACTUAL sampled parameters in this skewness bin
%            ⮕ Title shows MEAN parameters of THOSE 10
%            ⮕ and mean skewness of THOSE 10
%   Col 2 — Histogram of averaged sample in bin
%   Col 3 — Histogram of averaged null distribution
%   Col 4 — Null + observed t + p-value + rejection region
%
% All data are generated fresh using the SAME priors and centring as
% your main GPU permutation generator.

parallel.gpu.enableCUDAForwardCompatibility(true);
rng(cfg.random_seed);

%% ================= CONFIG =================
nSamples      = cfg.data.n_pvalues;
nParticipants = cfg.data.n_participants;
nPerms        = min(cfg.data.n_permutations, 40000);
BATCH         = min(5000, nSamples);

dists = {'LOGNORMAL', 'GAMMA', 'EXPONENTIAL'};

fprintf('\n=== GPU PERMUTATION DIAGNOSTICS (BATCHED) ===\n');
fprintf('Samples=%d  Participants=%d  Perms=%d  Batch=%d\n', ...
    nSamples, nParticipants, nPerms, BATCH);

%% ============== PERM SIGN MATRIX ==========
permSigns = 2 * (rand(nPerms, nParticipants) > 0.5) - 1;

if gpuDeviceCount > 0
    g = gpuDevice;
    fprintf('Using GPU: %s (%.1f GB free)\n', g.Name, g.AvailableMemory/1e9);
    Sg = gpuArray(permSigns);
    USE_GPU = true;
else
    fprintf('No GPU detected — CPU fallback.\n');
    Sg = permSigns;
    USE_GPU = false;
end


%% ===================================================================
%  LOOP DISTRIBUTIONS
%% ===================================================================
for d = 1:numel(dists)

    dist = dists{d};
    fprintf('\n==================== %s ====================\n', dist);

    %% --- STEP 1 — Generate samples & store params ---
    X = zeros(nSamples, nParticipants);

    shapeVec = []; scaleVec = [];
    rateVec  = [];
    muVec    = []; sigmaVec = [];

    switch dist
        case 'GAMMA'
            shapeVec = zeros(nSamples,1);
            scaleVec = zeros(nSamples,1);
        case 'EXPONENTIAL'
            rateVec  = zeros(nSamples,1);
        case 'LOGNORMAL'
            muVec    = zeros(nSamples,1);
            sigmaVec = zeros(nSamples,1);
    end

    for i = 1:nSamples
        [xi,p1,p2] = drawSampleWithParams(dist,nParticipants);
        X(i,:) = xi(:)';
        switch dist
            case 'GAMMA', shapeVec(i)=p1; scaleVec(i)=p2;
            case 'EXPONENTIAL', rateVec(i)=p1;
            case 'LOGNORMAL', muVec(i)=p1; sigmaVec(i)=p2;
        end
    end

    %% --- STEP 2 — Skewness + observed t ---
    skewVals = skewness(X,0,2);
    means = mean(X,2);
    stdev = std(X,0,2);
    obsT = means ./ (stdev ./ sqrt(nParticipants));

    %% --- STEP 3 — Skewness bins ---
    prc = linspace(2.5,97.5,10);
    edges = prctile(skewVals,prc);
    nBins = 9;

    binN = zeros(nBins,1);
    binAvgObs  = zeros(nBins,1);
    binAvgNull = zeros(nPerms,nBins);
    binAvgSamp = zeros(nBins,nParticipants);

    %% --- STEP 4 — GPU batched permutation nulls ---
    nBatches = ceil(nSamples/BATCH);
    for b = 1:BATCH:nSamples
        batchIdx = ceil(b/BATCH);
        fprintf('  Batch %d/%d...\n', batchIdx,nBatches);
        j = min(b+BATCH-1,nSamples);
        thisIdx = b:j;
        Bn = numel(thisIdx);

        Xb = X(thisIdx,:);
        Tob = obsT(thisIdx);
        skewB = skewVals(thisIdx);

        if USE_GPU
            Xg = gpuArray(Xb');
            permMeans = (Sg*Xg) ./ nParticipants;
            denom = sqrt(sum(Xg.^2,1)/(nParticipants^2));
            TnullB = permMeans ./ denom;
            TnullB = gather(TnullB);
        else
            Xt = Xb';
            permMeans = (permSigns*Xt)./nParticipants;
            denom = sqrt(sum(Xt.^2,1)/(nParticipants^2));
            TnullB = permMeans ./ denom;
        end

        for bb = 1:Bn
            s = skewB(bb);
            k = find(s>=edges(1:end-1)&s<=edges(2:end),1,'first');
            if isempty(k), continue; end
            binN(k)=binN(k)+1;
            a=1/binN(k);
            binAvgObs(k)=(1-a)*binAvgObs(k)+a*Tob(bb);
            binAvgSamp(k,:)=(1-a)*binAvgSamp(k,:)+a*Xb(bb,:);
            binAvgNull(:,k)=(1-a)*binAvgNull(:,k)+a*TnullB(:,bb);
        end
    end

    %% --- STEP 5 — Plotting ---
    figure('Position',[50,50,2300,2000]);

    for k = 1:nBins

        if binN(k)==0, continue; end

        avgSamp = binAvgSamp(k,:);
        avgNull = binAvgNull(:,k);
        avgObs  = binAvgObs(k);

        % indices in this bin
        if k<nBins
            idxBin = skewVals>=edges(k) & skewVals<edges(k+1);
        else
            idxBin = skewVals>=edges(k) & skewVals<=edges(k+1);
        end
        idxBinIdx = find(idxBin);
        nBin = numel(idxBinIdx);

        % choose up to 10 representative samples
        nExamples = min(10,nBin);
        exIdx = idxBinIdx(randperm(nBin,nExamples));

        % compute mean params of JUST THESE 10
        switch dist
            case 'GAMMA'
                meanShape = mean(shapeVec(exIdx));
                meanScale = mean(scaleVec(exIdx));
            case 'EXPONENTIAL'
                meanRate = mean(rateVec(exIdx));
            case 'LOGNORMAL'
                meanMu = mean(muVec(exIdx));
                meanSigma = mean(sigmaVec(exIdx));
        end

        % compute skewness of JUST THESE 10
        meanSkew10 = mean(skewVals(exIdx));

        % x-range for PDFs
        XbinVals = X(idxBin,:);
        xMin = prctile(XbinVals(:),1);
        xMax = prctile(XbinVals(:),99);
        xGrid = linspace(xMin,xMax,300);

        %% ===== Column 1 — PDFs =====
        subplot(nBins,4,(k-1)*4+1);
        hold on;
        for m = 1:nExamples
            ii = exIdx(m);
            switch dist
                case 'GAMMA'
                    y = pdfFromParams('GAMMA',xGrid,shapeVec(ii),scaleVec(ii));
                case 'EXPONENTIAL'
                    y = pdfFromParams('EXPONENTIAL',xGrid,rateVec(ii),NaN);
                case 'LOGNORMAL'
                    y = pdfFromParams('LOGNORMAL',xGrid,muVec(ii),sigmaVec(ii));
            end
            plot(xGrid,y,'LineWidth',1);
        end
        hold off; grid on;

        % TITLE now shows MEAN PARAMS OF THESE 10
        switch dist
            case 'GAMMA'
                ttl = sprintf(...
                    'GAMMA  shape=%.2f  scale=%.2f   skew=%.2f   n=%d',...
                    meanShape,meanScale,meanSkew10,nBin);
            case 'EXPONENTIAL'
                ttl = sprintf(...
                    'EXP  rate=%.2f   skew=%.2f   n=%d',...
                    meanRate,meanSkew10,nBin);
            case 'LOGNORMAL'
                ttl = sprintf(...
                    'LOGN  \\mu=%.2f  \\sigma=%.2f   skew=%.2f   n=%d',...
                    meanMu,meanSigma,meanSkew10,nBin);
        end
        title(ttl);

        xlabel('x'); ylabel('pdf');

        %% ===== Column 2 =====
        subplot(nBins,4,(k-1)*4+2);
        histogram(avgSamp,20,'FaceColor',[0.3 0.5 0.8],'EdgeColor','none');
        grid on;
        title('Averaged sample');

        %% ===== Column 3 =====
        subplot(nBins,4,(k-1)*4+3);
        histogram(avgNull,30,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none');
        grid on;
        title('Averaged null');

        %% ===== Column 4 =====
        subplot(nBins,4,(k-1)*4+4);
        histogram(avgNull,30,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none');
        hold on;
        crit = prctile(abs(avgNull),95);
        xl=xlim; yl=ylim;
        fill([xl(1) -crit -crit xl(1)],[0 0 yl(2) yl(2)],'r','FaceAlpha',0.15,'EdgeColor','none');
        fill([crit xl(2) xl(2) crit],[0 0 yl(2) yl(2)],'r','FaceAlpha',0.15,'EdgeColor','none');
        xline(avgObs,'r','LineWidth',2);
        p=(sum(abs(avgNull)>=abs(avgObs))+1)/(numel(avgNull)+1);
        title(sprintf('p=%.4f  t=%.2f',p,avgObs));
        grid on; hold off;
    end

    sgtitle(sprintf('%s — permutation diagnostics by skewness bin',dist));
end

fprintf('\n=== COMPLETE ===\n');

end


%% ===============================================================
% SAMPLE + PARAMS (MATCH PRIORS)
%% ===============================================================
function [x,p1,p2] = drawSampleWithParams(dist,n)

switch upper(dist)
    case 'GAMMA'
        shape = 0.2 + rand()*(2.0-0.2);
        scale = 0.5 + rand()*(4.0-0.5);
        x = gamrnd(shape,scale,n,1) - shape*scale;
        p1=shape; p2=scale;

    case 'EXPONENTIAL'
        rate = 0.3 + rand()*(2.5-0.3);
        x = exprnd(1/rate,n,1) - 1/rate;
        p1=rate; p2=NaN;

    case 'LOGNORMAL'
        mu=-0.5 + rand()*(0.5-(-0.5));
        sigma=0.5 + rand()*(0.9-0.5);
        m=exp(mu+sigma^2/2);
        x = lognrnd(mu,sigma,n,1) - m;
        p1=mu; p2=sigma;

    otherwise
        error('Unknown distribution');
end

end


%% ===============================================================
% PDF (ON CENTRED SCALE)
%% ===============================================================
function y = pdfFromParams(dist,x,p1,p2)

y=zeros(size(x));

switch upper(dist)
    case 'GAMMA'
        m=p1*p2;
        z=x+m;
        mask=z>0;
        y(mask)=gampdf(z(mask),p1,p2);

    case 'EXPONENTIAL'
        m=1/p1;
        z=x+m;
        mask=z>0;
        y(mask)=p1.*exp(-p1.*z(mask));

    case 'LOGNORMAL'
        m=exp(p1+p2^2/2);
        z=x+m;
        mask=z>0;
        y(mask)=lognpdf(z(mask),p1,p2);
end

end
