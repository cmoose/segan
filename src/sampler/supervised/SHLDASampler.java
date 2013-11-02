package sampler.supervised;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirMult;
import sampling.likelihood.TruncatedStickBreaking;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.StatisticsUtils;
import util.evaluation.Measurement;
import util.evaluation.MimnoTopicCoherence;
import util.evaluation.RegressionEvaluation;

/**
 * Gibbs sampler for Original SHLDA where each document is assigned to a path in
 * the topic tree.
 *
 * @author vietan
 */
public class SHLDASampler extends AbstractSampler {

    public static final int PSEUDO_NODE_INDEX = -1;
    public static final int RHO = 0;
    public static final int MEAN = 1;
    public static final int SCALE = 2;
    protected boolean supervised = true;
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected double[] mus;
    protected double[] sigmas;
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int regressionLevel;
    protected int[][] words;  // words
    protected double[] responses; // [D]: document observations
    private int[][] z; // level assignments
    private SHLDANode[] c; // path assignments
    private TruncatedStickBreaking[] doc_level_distr;
    private SHLDANode word_hier_root;
    private DirMult[] emptyModels;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private double[] uniform;
    private int numChangePath;
    private int numChangeLevel;
    private int optimizeCount = 0;
    private int convergeCount = 0;
    private int numTokens = 0;

    public void configure(String folder, int[][] words, double[] y,
            int V, int L,
            double mean, // GEM mean
            double scale, // GEM scale
            double[] betas, // Dirichlet hyperparameter for distributions over words
            double[] gammas, // hyperparameter for nCRP
            double[] mus, // mean of Gaussian for regression parameters
            double[] sigmas, // stadard deviation of Gaussian for regression parameters
            double rho, // standard deviation of Gaussian for document observations
            int regressLevel,
            AbstractSampler.InitialState initState, boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInterval) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.words = words;
        this.responses = y;
        if (this.responses == null) {
            this.supervised = false;
        }

        this.L = L;
        this.V = V;
        this.D = this.words.length;

        this.betas = betas;
        this.gammas = gammas;
        this.mus = mus;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(rho);
        this.hyperparams.add(mean);
        this.hyperparams.add(scale);

        for (int l = 0; l < betas.length; l++) {
            this.hyperparams.add(betas[l]);
        }

        for (int i = 0; i < gammas.length; i++) {
            this.hyperparams.add(gammas[i]);
        }

        for (int i = 0; i < mus.length; i++) {
            this.hyperparams.add(mus[i]);
        }

        for (int i = 0; i < sigmas.length; i++) {
            this.hyperparams.add(sigmas[i]);
        }

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInterval;
        this.regressionLevel = regressLevel;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        this.uniform = new double[V];
        for (int i = 0; i < V; i++) {
            uniform[i] = 1.0 / V;
        }

        // assert dimensions
        if (this.betas.length != this.L) {
            throw new RuntimeException("Vector betas must have length " + this.L
                    + ". Current length = " + this.betas.length);
        }
        if (this.gammas.length != this.L - 1) {
            throw new RuntimeException("Vector gamms must have length " + (this.L - 1)
                    + ". Current length = " + this.gammas.length);
        }
        if (this.mus.length != this.L) {
            throw new RuntimeException("Vector mus must have length " + this.L
                    + ". Current length = " + this.mus.length);
        }
        if (this.sigmas.length != this.L) {
            throw new RuntimeException("Vector sigmas must have length " + this.L
                    + ". Current length = " + this.sigmas.length);
        }

        if (!debug) {
            System.err.close();
        }

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- max level:\t" + L);
            logln("--- reg level:\t" + regressLevel);
            logln("--- GEM mean:\t" + hyperparams.get(MEAN));
            logln("--- GEM scale:\t" + hyperparams.get(SCALE));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- reg mus:\t" + MiscUtils.arrayToString(mus));
            logln("--- reg sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- response rho:\t" + hyperparams.get(RHO));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_SHLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_LVL-").append(L)
                .append("_RL-").append(regressionLevel)
                .append("_GEM-M-").append(formatter.format(hyperparams.get(MEAN)))
                .append("_GEM-S-").append(formatter.format(hyperparams.get(SCALE)))
                .append("_RHO-").append(formatter.format(hyperparams.get(RHO)));

        int count = SCALE + 1;
        str.append("_b");
        for (int i = 0; i < betas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_g");
        for (int i = 0; i < gammas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_m");
        for (int i = 0; i < mus.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_s");
        for (int i = 0; i < sigmas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }

        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        iter = INIT;

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }

    protected void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing topic hierarchy ...");
        }

//        HashMap<Integer, Integer> tfs = new HashMap<Integer, Integer>();
//        HashMap<Integer, Integer> dfs = new HashMap<Integer, Integer>();
//        HashMap<Integer, Double> sumRes = new HashMap<Integer, Double>();
//        for(int d=0; d<D; d++){
//            Set<Integer> docUniqueTerms = new HashSet<Integer>();
//            for(int n=0; n<words[d].length; n++){
//                int token = words[d][n];
//                docUniqueTerms.add(token);
//                
//                Integer tf = tfs.get(token);
//                if(tf == null)
//                    tfs.put(token, 1);
//                else
//                    tfs.put(token, tf + 1);
//            }
//            
//            for(int token : docUniqueTerms){
//                Integer df = dfs.get(token);
//                if(df == null)
//                    dfs.put(token, 1);
//                else
//                    dfs.put(token, df + 1);
//                
//                Double sr = sumRes.get(token);
//                if(sr == null)
//                    sumRes.put(token, responses[d]);
//                else
//                    sumRes.put(token, sr + responses[d]);
//            }
//        }
//        
//        int maxTf = 0;
//        for(int type : tfs.keySet()){
//            if(maxTf < tfs.get(type))
//                maxTf = tfs.get(type);
//        }
//        
//        double[] rootMean = new double[V];
//        for(int v=0; v<V; v++){
//            if(tfs.containsKey(v)){
//                double tf = 0.5 + 0.5 * tfs.get(v) / maxTf;
//                double idf = Math.log(D) - Math.log(dfs.get(v));
//                double tf_idf = tf * idf;
//                double sent = sumRes.get(v);
//
//                double value = 1.0 / (tf_idf * Math.abs(sent));
//                rootMean[v] = value * betas[0];
//            }
//            else{
//                rootMean[v] = 1.0 / V * betas[0];
//            }
//        }

        int rootLevel = 0;
        int rootIndex = 0;
        DirMult dmModel = new DirMult(V, betas[rootLevel], uniform);
        double regParam = SamplerUtils.getGaussian(mus[rootLevel], sigmas[rootLevel]);
        this.word_hier_root = new SHLDANode(iter, rootIndex, rootLevel, dmModel, regParam, null);

        this.emptyModels = new DirMult[L - 1];
        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l] = new DirMult(V, betas[l + 1], uniform);
        }
    }

    protected void initializeDataStructure() {
        c = new SHLDANode[D];
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        this.doc_level_distr = new TruncatedStickBreaking[D];
        for (int d = 0; d < D; d++) {
            this.doc_level_distr[d] = new TruncatedStickBreaking(L, hyperparams.get(MEAN), hyperparams.get(SCALE));
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }

        if (verbose) {
            logln("--- Done initialization. "
                    + "Llh = " + this.getLogLikelihood()
                    + "\t" + this.getCurrentState());
        }
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        // initialize path assignments
        for (int d = 0; d < D; d++) {
            SHLDANode node = word_hier_root;
            for (int l = 0; l < L - 1; l++) {
                node.incrementNumCustomers();
                node = this.createNode(node); // create a new path for each document
            }
            node.incrementNumCustomers();
            c[d] = node;

            // forward sample levels
            for (int n = 0; n < words[d].length; n++) {
                sampleLevelAssignments(d, n, !REMOVE, ADD, !REMOVE, ADD, OBSERVED);
            }

            // resample path
            if (d > 0) {
                samplePathAssignments(d, REMOVE, ADD, OBSERVED, EXTEND);
            }

            // resampler levels
            for (int n = 0; n < words[d].length; n++) {
                sampleLevelAssignments(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        try {
            if (report) {
                IOUtils.createFolder(this.folder + this.getSamplerFolder() + ReportFolder);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose && iter % REP_INTERVAL == 0) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t" + getCurrentState());
                }
            }

            numChangePath = 0;
            numChangeLevel = 0;
            optimizeCount = 0;
            convergeCount = 0;
            numChangePath = 0;
            numChangeLevel = 0;

            for (int d = 0; d < D; d++) {
                samplePathAssignments(d, REMOVE, ADD, OBSERVED, EXTEND);

                for (int n = 0; n < words[d].length; n++) {
                    sampleLevelAssignments(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
                }
            }

            if (supervised) {
                updateRegressionParameters();
            }

            if (verbose && supervised && iter % REP_INTERVAL == 0) {
                double[] trPredResponses = getRegressionValues();
                RegressionEvaluation eval = new RegressionEvaluation(
                        (responses),
                        (trPredResponses));
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
            }

            if (debug) {
                validate("iter " + iter);
            }

            if (iter % LAG == 0 && iter >= BURN_IN) {
                if (paramOptimized) { // slice sampling
                    if (verbose) {
                        logln("*** *** Optimizing hyperparameters by slice sampling ...");
                        logln("*** *** cur param:" + MiscUtils.listToString(hyperparams));
                        logln("*** *** new llh = " + this.getLogLikelihood());
                    }

                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        logln("*** *** new param:" + MiscUtils.listToString(sparams));
                        logln("*** *** new llh = " + this.getLogLikelihood());
                    }
                }
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(this.folder + this.getSamplerFolder() + ReportFolder + "iter-" + iter + ".zip");
                try {
                    outputTopicTopWords(this.folder + this.getSamplerFolder() + ReportFolder + "iter-" + iter + "-top-words.txt", 15);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(1);
                }
            }
        }

        if (report) {
            outputState(this.folder + this.getSamplerFolder() + "final.zip");
            outputState(this.folder + this.getSamplerFolder() + ReportFolder + "iter-" + iter + ".zip");
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(this.folder + this.getSamplerFolder() + "likelihoods.txt");
            for (int i = 0; i < logLikelihoods.size(); i++) {
                writer.write(i + "\t" + logLikelihoods.get(i) + "\n");
            }
            writer.close();

            if (paramOptimized && log) {
                this.outputSampledHyperparameters(this.folder + this.getSamplerFolder() + "hyperparameters.txt");
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private double[] getRegressionValuesWithoutRoot() {
        double[] regValues = new double[D];
        for (int d = 0; d < D; d++) {
            double[] regParams = getRegressionPath(c[d]);

            double sum = 0.0;
            int count = 0;
            for (int l = 1; l < L; l++) {
                sum += regParams[l] * doc_level_distr[d].getCount(l);
                count += doc_level_distr[d].getCount(l);
            }

            if (count == 0) {
                regValues[d] = 0.0;
            } else {
                regValues[d] = sum / count;
            }
        }
        return regValues;
    }

    private double[] getRegressionValues() {
        double[] regValues = new double[D];
        for (int d = 0; d < D; d++) {
            double[] regParams = getRegressionPath(c[d]);
            double sum = 0.0;
            for (int l = 0; l < L; l++) {
                sum += regParams[l] * doc_level_distr[d].getCount(l);
            }
            regValues[d] = sum / words[d].length;
        }
        return regValues;
    }

    /**
     * Add a customer to a path. A path is specified by the pointer to its leaf
     * node. If the given node is not a leaf node, an exception will be thrown.
     * The number of customers at each node on the path will be incremented.
     *
     * @param leafNode The leaf node of the path
     */
    private void addCustomerToPath(SHLDANode leafNode) {
        SHLDANode node = leafNode;
        while (node != null) {
            node.incrementNumCustomers();
            node = node.getParent();
        }
    }

    /**
     * Remove a customer from a path. A path is specified by the pointer to its
     * leaf node. If the given node is not a leaf node, an exception will be
     * thrown. The number of customers at each node on the path will be
     * decremented. If the number of customers at a node is 0, the node will be
     * removed.
     *
     * @param leafNode The leaf node of the path
     * @return Return the node that specifies the path that the leaf node is
     * removed from. If a lower-level node has no customer, it will be removed
     * and the lowest parent node on the path that has non-zero number of
     * customers will be returned.
     */
    private SHLDANode removeCustomerFromPath(SHLDANode leafNode) {
        SHLDANode retNode = leafNode;
        SHLDANode node = leafNode;
        while (node != null) {
            node.decrementNumCustomers();
            if (node.isEmpty()) {
                retNode = node.getParent();
                node.getParent().removeChild(node.getIndex());
            }
            node = node.getParent();
        }
        return retNode;
    }

    /**
     * Remove an observation from a node.
     *
     * @param observation The observation to be added
     * @param level The level of the node
     * @param leafNode The leaf node of the path
     */
    private void removeObservation(int observation, int level, SHLDANode leafNode) {
        SHLDANode node = getNode(level, leafNode);
        node.getContent().decrement(observation);
    }

    private void removeObservationsFromPath(SHLDANode leafNode, HashMap<Integer, Integer>[] observations) {
        SHLDANode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            removeObservationsFromNode(path[l], observations[l]);
        }
    }

    private void removeObservationsFromNode(SHLDANode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, -count);
        }
    }

    private void addObservationsToPath(SHLDANode leafNode, HashMap<Integer, Integer>[] observations) {
        SHLDANode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            addObservationsToNode(path[l], observations[l]);
        }
    }

    private void addObservationsToNode(SHLDANode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, count);
        }
    }

    /**
     * Add an observation to a node
     *
     * @param observation The observation to be added
     * @param level The level of the node
     * @param leafNode The leaf node of the path
     */
    private void addObservation(int observation, int level, SHLDANode leafNode) {
        SHLDANode node = getNode(level, leafNode);
        node.getContent().increment(observation);
    }

    /**
     * Create a new child of a parent node
     *
     * @param parent The parent node
     * @return The newly created child node
     */
    private SHLDANode createNode(SHLDANode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirMult dmModel = new DirMult(V, betas[level], uniform);
        double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
        SHLDANode child = new SHLDANode(iter, nextChildIndex, level, dmModel, regParam, parent);
        return parent.addChild(nextChildIndex, child);
    }

    private void samplePathAssignments(int d, boolean remove, boolean add, boolean observed, boolean extend) {
        SHLDANode curPath = c[d];

        HashMap<Integer, Integer>[] docTypeCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            docTypeCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int n = 0; n < words[d].length; n++) {
            Integer count = docTypeCountPerLevel[z[d][n]].get(words[d][n]);
            if (count == null) {
                docTypeCountPerLevel[z[d][n]].put(words[d][n], 1);
            } else {
                docTypeCountPerLevel[z[d][n]].put(words[d][n], count + 1);
            }
        }

        double[] dataLlhNewTopic = new double[L];
        for (int l = 1; l < L; l++) // skip the root
        {
            dataLlhNewTopic[l] = emptyModels[l - 1].getLogLikelihood(docTypeCountPerLevel[l]);
        }

        if (remove) {
            removeObservationsFromPath(c[d], docTypeCountPerLevel);
            removeCustomerFromPath(c[d]);
        }

        HashMap<SHLDANode, Double> pathLogPriors = new HashMap<SHLDANode, Double>();
        computePathLogPrior(pathLogPriors, word_hier_root, 0.0);

        HashMap<SHLDANode, Double> pathWordLlhs = new HashMap<SHLDANode, Double>();
        computePathWordLogLikelihood(pathWordLlhs, word_hier_root,
                docTypeCountPerLevel, dataLlhNewTopic, 0.0);

        if (pathLogPriors.size() != pathWordLlhs.size()) {
            throw new RuntimeException("Numbers of paths mismatch");
        }

        HashMap<SHLDANode, Double> pathResLlhs = new HashMap<SHLDANode, Double>();
        if (supervised && observed) {
            pathResLlhs = computePathResponseLogLikelihood(d);

            if (pathLogPriors.size() != pathResLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch");
            }
        }

        // sample path
        ArrayList<Double> logprobs = new ArrayList<Double>();
        ArrayList<SHLDANode> pathList = new ArrayList<SHLDANode>();
        for (SHLDANode path : pathLogPriors.keySet()) {
            if (!extend && !isLeafNode(path)) // during test time, fix the tree
            {
                continue;
            }

            double logPrior = pathLogPriors.get(path);
            double wordLlh = pathWordLlhs.get(path);
            double lp = logPrior + wordLlh;

            // debug
//            logln(d + ". " + words[d].length
//                    + ". log prior = " + MiscUtils.formatDouble(logPrior)
//                    + ". word llh = " + MiscUtils.formatDouble(wordLlh)
//                    + ". path = " + path.toString());

            if (supervised && observed) {
                lp += pathResLlhs.get(path);
            }
            pathList.add(path);
            logprobs.add(lp);
        }

        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);

        if (sampledIndex == logprobs.size()) {
            logln(MiscUtils.listToString(logprobs));
            throw new RuntimeException("Out-of-bound scale sampling");
        }

        SHLDANode newPath = pathList.get(sampledIndex);

        if (curPath != null && curPath.getIndex() != sampledIndex) {
            numChangePath++;
        }

        if (newPath.getLevel() < L - 1) {
            newPath = this.createNewPath(newPath);
        }

        c[d] = newPath;

        if (add) {
            addCustomerToPath(c[d]);
            addObservationsToPath(c[d], docTypeCountPerLevel);
        }
    }

    /**
     * Sample the level assignment z[d][n] for a token.
     *
     * @param d The document index
     * @param n The word position in the document
     * @param remove Whether this token should be removed from the current state
     */
    private void sampleLevelAssignments(
            int d,
            int n,
            boolean removeLevelDist,
            boolean addLevelDist,
            boolean removeWordHier,
            boolean addWordHier,
            boolean observed) {
        // decrement 
        if (removeLevelDist) {
            doc_level_distr[d].decrement(z[d][n]);
        }
        if (removeWordHier) {
            removeObservation(words[d][n], z[d][n], c[d]);
        }

        double[] pathRegParams = this.getRegressionPath(c[d]);
        double preSum = 0.0;
        for (int l = 0; l < L; l++) {
            preSum += pathRegParams[l] * doc_level_distr[d].getCount(l);
        }

        double[] logprobs = new double[L];
        for (int l = 0; l < L; l++) {
            // sampling equation
            SHLDANode node = this.getNode(l, c[d]);
            double logprior = doc_level_distr[d].getLogProbability(l);
//            double logprior = doc_level_distr[d].getLogLikelihood(l);
            double wordLlh = node.getContent().getLogLikelihood(words[d][n]);

            logprobs[l] = logprior + wordLlh;

            if (observed) {
                double sum = preSum + pathRegParams[l];
                double mean = sum / words[d].length;
                double resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, Math.sqrt(hyperparams.get(RHO)));
                logprobs[l] += resLlh;

                // debug
//                logln("iter = " + iter 
//                        + ". d = " + d
//                        + ". n = " + n
//                        + ". l = " + l
//                        + ". log prior = " + MiscUtils.formatDouble(logprior)
//                        + ". word llh = " + MiscUtils.formatDouble(wordLlh)
//                        + ". res llh = " + MiscUtils.formatDouble(resLlh)
//                        + ". token: " + wordVocab.get(words[d][n])
//                        + ". empty topic: " + MiscUtils.formatDouble(emptyModels[0].getLogLikelihood(words[d][n]))
//                        );
            }
        }

        int sampledL = SamplerUtils.logMaxRescaleSample(logprobs);
        if (z[d][n] != sampledL) {
            numChangeLevel++;
        }

        // debug
//        logln("---> sampled l = " + sampledL + "\n");

        // update and increment
        z[d][n] = sampledL;

        if (addLevelDist) {
            doc_level_distr[d].increment(z[d][n]);
        }
        if (addWordHier) {
            this.addObservation(words[d][n], z[d][n], c[d]);
        }
    }

    private void updateRegressionParameters() {
        Queue<SHLDANode> queue = new LinkedList<SHLDANode>();
        queue.add(word_hier_root);
        while (!queue.isEmpty()) {
            SHLDANode node = queue.poll();
            // update for all subtrees having the root node at the regressLevel
            if (node.getLevel() == regressionLevel) {
                optimize(node);
            }

            // recurse
            for (SHLDANode child : node.getChildren()) {
                queue.add(child);
            }
        }
    }

    private void optimize(SHLDANode root) {
        ArrayList<SHLDANode> flatTree = this.flattenTree(root);

        // map a node in the subtree to its index in the flattened tree (i.e. 1D array)
        HashMap<SHLDANode, Integer> nodeMap = new HashMap<SHLDANode, Integer>();
        for (int i = 0; i < flatTree.size(); i++) {
            nodeMap.put(flatTree.get(i), i);
        }

        ArrayList<double[]> designList = new ArrayList<double[]>();
        ArrayList<Double> responseList = new ArrayList<Double>();
        for (int d = 0; d < D; d++) {
            if (!nodeMap.containsKey(c[d])) // if this subtree does not contain this document
            {
                continue;
            }
            SHLDANode[] docPath = this.getPathFromNode(c[d]);
            double[] empiricalProb = doc_level_distr[d].getEmpiricalDistribution();

            double[] row = new double[flatTree.size()];
            for (int l = root.getLevel(); l < L; l++) {
                int nodeIndex = nodeMap.get(docPath[l]);
                row[nodeIndex] = empiricalProb[l];
            }
            designList.add(row);
            responseList.add(responses[d]);
        }

        double[][] designMatrix = new double[designList.size()][flatTree.size()];
        for (int i = 0; i < designList.size(); i++) {
            designMatrix[i] = designList.get(i);
        }

        double[] response = new double[responseList.size()];
        for (int i = 0; i < response.length; i++) {
            response[i] = responseList.get(i);
        }

        double[] curParams = new double[flatTree.size()];
        for (int i = 0; i < flatTree.size(); i++) {
            curParams[i] = flatTree.get(i).getRegressionParameter();
        }

        // get the arrays of mus and sigmas corresponding to the list of flatten nodes
        double[] flattenMus = new double[flatTree.size()];
        double[] flattenSigmas = new double[flatTree.size()];
        for (int i = 0; i < flatTree.size(); i++) {
            int nodeLevel = flatTree.get(i).getLevel();
            flattenMus[i] = this.mus[nodeLevel];
            flattenSigmas[i] = this.sigmas[nodeLevel];
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                curParams, designMatrix, response,
                Math.sqrt(hyperparams.get(RHO)),
                flattenMus, flattenSigmas);
        this.optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            // This exception may be thrown if L-BFGS
            //  cannot step in the current direction.
            // This condition does not necessarily mean that
            //  the optimizer has failed, but it doesn't want
            //  to claim to have succeeded... 
            // do nothing
        }

        optimizeCount++;

        // if the number of observations is less than or equal to the number of parameters
        if (converged) {
            convergeCount++;
        }

        // update regression parameters
        for (int i = 0; i < flatTree.size(); i++) {
            flatTree.get(i).setRegressionParameter(optimizable.getParameter(i));
        }
    }

    private void computePathLogPrior(
            HashMap<SHLDANode, Double> nodeLogProbs,
            SHLDANode curNode,
            double parentLogProb) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            double logNorm = Math.log(curNode.getNumCustomers() + gammas[curNode.getLevel()]);
            newWeight += Math.log(gammas[curNode.getLevel()]) - logNorm;

            for (SHLDANode child : curNode.getChildren()) {
                double childWeight = parentLogProb + Math.log(child.getNumCustomers()) - logNorm;
                computePathLogPrior(nodeLogProbs, child, childWeight);
            }

        }
        nodeLogProbs.put(curNode, newWeight);
    }

    private void computePathWordLogLikelihood(
            HashMap<SHLDANode, Double> nodeDataLlhs,
            SHLDANode curNode,
            HashMap<Integer, Integer>[] docTokenCountPerLevel,
            double[] dataLlhNewTopic,
            double parentDataLlh) {

        int level = curNode.getLevel();
        double nodeDataLlh = curNode.getContent().getLogLikelihood(docTokenCountPerLevel[level]);

//        logln("--- " + curNode.toString() + ". " + MiscUtils.formatDouble(nodeDataLlh)
//                + ". " + docTokenCountPerLevel[level].size());
//        if(nodeDataLlh < -1000){
//            System.out.println(docTokenCountPerLevel[level].toString());
//        }

        // populate to child nodes
        for (SHLDANode child : curNode.getChildren()) {
            computePathWordLogLikelihood(nodeDataLlhs, child, docTokenCountPerLevel,
                    dataLlhNewTopic, parentDataLlh + nodeDataLlh);
        }

        // store the data llh from the root to this current node
        double storeDataLlh = parentDataLlh + nodeDataLlh;
        level++;
        while (level < L) // if this is an internal node, add llh of new child node
        {
            storeDataLlh += dataLlhNewTopic[level++];
        }
        nodeDataLlhs.put(curNode, storeDataLlh);
    }

    private HashMap<SHLDANode, Double> computePathResponseLogLikelihood(int d) {
        HashMap<SHLDANode, Double> resLlhs = new HashMap<SHLDANode, Double>();
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();
            SHLDANode[] path = getPathFromNode(node);
            double sum = 0.0;
            double var = hyperparams.get(RHO);
            int level;
            for (level = 0; level < path.length; level++) {
                sum += path[level].getRegressionParameter() * doc_level_distr[d].getCount(level);
            }
            while (level < L) {
                int levelCount = doc_level_distr[d].getCount(level);
                sum += levelCount * mus[level];
                var += Math.pow((double) levelCount / words[d].length, 2) * sigmas[level];
                level++;
            }

            double mean = sum / words[d].length;
            double resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
            resLlhs.put(node, resLlh);

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return resLlhs;
    }

    private boolean isLeafNode(SHLDANode node) {
        return node.getLevel() == L - 1;
    }

    private SHLDANode createNewPath(SHLDANode internalNode) {
        SHLDANode node = internalNode;
        for (int l = internalNode.getLevel(); l < L - 1; l++) {
            node = this.createNode(node);
        }
        return node;
    }

    /**
     * Get a node at a given level on a path on the tree. The path is determined
     * by its leaf node.
     *
     * @param level The level that the node is at
     * @param leafNode The leaf node of the path
     */
    private SHLDANode getNode(int level, SHLDANode leafNode) {
        if (!isLeafNode(leafNode)) {
            throw new RuntimeException("Exception while getting node. The given "
                    + "node is not a leaf node");
        }
        int curLevel = leafNode.getLevel();
        SHLDANode curNode = leafNode;
        while (curLevel != level) {
            curNode = curNode.getParent();
            curLevel--;
        }
        return curNode;
    }

    /**
     * Flatten a subtree given the root of the subtree
     *
     * @param subtreeRoot The subtree's root
     */
    private ArrayList<SHLDANode> flattenTree(SHLDANode subtreeRoot) {
        ArrayList<SHLDANode> flatSubtree = new ArrayList<SHLDANode>();
        Queue<SHLDANode> queue = new LinkedList<SHLDANode>();
        queue.add(subtreeRoot);
        while (!queue.isEmpty()) {
            SHLDANode node = queue.poll();
            flatSubtree.add(node);
            for (SHLDANode child : node.getChildren()) {
                queue.add(child);
            }
        }
        return flatSubtree;
    }

    /**
     * Return a path from the root to a given node
     *
     * @param node The given node
     * @return An array containing the path
     */
    private SHLDANode[] getPathFromNode(SHLDANode node) {
        SHLDANode[] path = new SHLDANode[node.getLevel() + 1];
        SHLDANode curNode = node;
        int l = node.getLevel();
        while (curNode != null) {
            path[l--] = curNode;
            curNode = curNode.getParent();
        }
        return path;
    }

    /**
     * Get an array containing all the regression parameters along a path. The
     * path is specified by a leaf node.
     *
     * @param leafNode The leaf node
     */
    private double[] getRegressionPath(SHLDANode leafNode) {
        if (leafNode.getLevel() != L - 1) {
            throw new RuntimeException("Node " + leafNode.toString() + " is not a leaf node");
        }
        double[] regPath = new double[L];
        int level = leafNode.getLevel();
        SHLDANode curNode = leafNode;
        while (curNode != null) {
            regPath[level--] = curNode.getRegressionParameter();
            curNode = curNode.getParent();
        }
        return regPath;
    }

    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    private SHLDANode getNode(int[] parsedPath) {
        SHLDANode node = word_hier_root;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    public String printGlobalTree() {
        StringBuilder str = new StringBuilder();
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);

        int totalObs = 0;

        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString())
                    .append("\n");

            totalObs += node.getContent().getCountSum();

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> # observations = ").append(totalObs)
                .append("\n>>> # customers = ").append(word_hier_root.getNumCustomers())
                .append("\n");
        return str.toString();
    }

    @Override
    public String getCurrentState() {
        int[] custCountPerLevel = new int[L];
        int[] obsCountPerLevel = new int[L];

        Queue<SHLDANode> queue = new LinkedList<SHLDANode>();
        queue.add(word_hier_root);
        while (!queue.isEmpty()) {
            SHLDANode node = queue.poll();
            custCountPerLevel[node.getLevel()]++;
            obsCountPerLevel[node.getLevel()] += node.getContent().getCountSum();

            // add children to the queue
            for (SHLDANode child : node.getChildren()) {
                queue.add(child);
            }
        }

        StringBuilder str = new StringBuilder();
        for (int l = 0; l < L; l++) {
            str.append(l).append("(")
                    .append(custCountPerLevel[l])
                    .append(", ").append(obsCountPerLevel[l])
                    .append(")\t");
        }
        str.append("total obs: ").append(StatisticsUtils.sum(obsCountPerLevel));
        return str.toString();
    }

    public void outputTopicTopWords(String outputFile, int numWords)
            throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing top words to file " + outputFile);
        }

        StringBuilder str = new StringBuilder();
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }

            // skip leaf nodes that are empty
            if (isLeafNode(node) && node.getContent().getCountSum() == 0) {
                continue;
            }
            if (node.getIterationCreated() >= MAX_ITER - LAG) {
                continue;
            }

            String[] topWords = getTopWords(node.getContent().getDistribution(), numWords);
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.getIterationCreated())
                    .append("; ").append(node.getNumCustomers())
                    .append("; ").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.getRegressionParameter()))
                    .append(")");
            for (String topWord : topWords) {
                str.append(" ").append(topWord);
            }
            str.append("\n\n");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        writer.write(str.toString());
        writer.close();
    }

    public void outputTopicCoherence(
            String filepath,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing topic coherence to file " + filepath);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);

        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
//        HashMap<SHLDANode, Double> scoreMap = new HashMap<SHLDANode, Double>();
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getContent().getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(node.getPathString()
                    + "\t" + node.getIterationCreated()
                    + "\t" + node.getNumCustomers()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");

//            if(node.getLevel() == 1){
//                scoreMap.put(node, score);
//            }
        }

        writer.close();

//        double sum = 0.0;
//        int count = 0;
//        for(SHLDANode node : scoreMap.keySet()){
//            int cs = node.getContent().getCountSum();
//            sum += scoreMap.get(node) * cs;
//            count += cs;
//        }
//        logln("weighted avg = " + MiscUtils.formatDouble(sum / count));
    }

    public void outputTopicWordDistributions(String filepath) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);

        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getContent().getDistribution();
            writer.write(node.getPathString());
            for (int v = 0; v < distribution.length; v++) {
                writer.write("\t" + distribution[v]);
            }
            writer.write("\n");
        }

        writer.close();
    }

    public void outputDocPathAssignments(String filepath) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int d = 0; d < D; d++) {
            SHLDANode docNode = this.c[d];
            writer.write(Integer.toString(d)
                    + "\t" + docNode.getPathString()
                    + "\n");
        }
        writer.close();
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        double treeLogProb = 0.0;
        double regParamLgprob = 0.0;
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood();

            if (supervised) {
                regParamLgprob += StatisticsUtils.logNormalProbability(node.getRegressionParameter(),
                        mus[node.getLevel()], Math.sqrt(sigmas[node.getLevel()]));
            }

            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(gammas[node.getLevel()]);
            }

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double stickLgprob = 0.0;
        double resLlh = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            stickLgprob += doc_level_distr[d].getLogLikelihood();
            if (supervised) {
                resLlh += StatisticsUtils.logNormalProbability(responses[d], regValues[d], Math.sqrt(hyperparams.get(RHO)));
            }
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("^^^ word-llh = " + MiscUtils.formatDouble(wordLlh)
                    + ". tree = " + MiscUtils.formatDouble(treeLogProb)
                    + ". stick = " + MiscUtils.formatDouble(stickLgprob)
                    + ". reg param = " + MiscUtils.formatDouble(regParamLgprob)
                    + ". response = " + MiscUtils.formatDouble(resLlh));
        }

        double llh = wordLlh + treeLogProb + stickLgprob + regParamLgprob + resLlh;
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        int count = SCALE + 1;
        double[] newBetas = new double[betas.length];
        for (int i = 0; i < newBetas.length; i++) {
            newBetas[i] = tParams.get(count++);
        }
        double[] newGammas = new double[gammas.length];
        for (int i = 0; i < newGammas.length; i++) {
            newGammas[i] = tParams.get(count++);
        }
        double[] newMus = new double[mus.length];
        for (int i = 0; i < newMus.length; i++) {
            newMus[i] = tParams.get(count++);
        }
        double[] newSigmas = new double[sigmas.length];
        for (int i = 0; i < newSigmas.length; i++) {
            newSigmas[i] = tParams.get(count++);
        }

        double wordLlh = 0.0;
        double treeLogProb = 0.0;
        double regParamLgprob = 0.0;
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood(newBetas[node.getLevel()], uniform);

            if (supervised) {
                regParamLgprob += StatisticsUtils.logNormalProbability(node.getRegressionParameter(),
                        newMus[node.getLevel()], Math.sqrt(newSigmas[node.getLevel()]));
            }

            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(newGammas[node.getLevel()]);
            }

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double stickLgprob = 0.0;
        double resLlh = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            stickLgprob += doc_level_distr[d].getLogLikelihood(tParams.get(MEAN), tParams.get(SCALE));
            if (supervised) {
                resLlh += StatisticsUtils.logNormalProbability(responses[d], regValues[d], Math.sqrt(tParams.get(RHO)));
            }
        }
        double llh = wordLlh + treeLogProb + stickLgprob + regParamLgprob + resLlh;
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> tParams) {
        this.hyperparams = new ArrayList<Double>();
        for (double param : tParams) {
            this.hyperparams.add(param);
        }

        int count = SCALE + 1;
        betas = new double[betas.length];
        for (int i = 0; i < betas.length; i++) {
            betas[i] = tParams.get(count++);
        }
        gammas = new double[gammas.length];
        for (int i = 0; i < gammas.length; i++) {
            gammas[i] = tParams.get(count++);
        }
        mus = new double[mus.length];
        for (int i = 0; i < mus.length; i++) {
            mus[i] = tParams.get(count++);
        }
        sigmas = new double[sigmas.length];
        for (int i = 0; i < sigmas.length; i++) {
            sigmas[i] = tParams.get(count++);
        }

        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();
            node.getContent().setConcentration(betas[node.getLevel()]);
            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l].setConcentration(betas[l + 1]);
        }

//        for(int d=0; d<D; d++){
//            doc_level_distr[d].setMean(hyperparams.get(MEAN));
//            doc_level_distr[d].setScale(hyperparams.get(SCALE));
//        }
    }

    @Override
    public void validate(String msg) {
        validateModel(msg);

        validateAssignments(msg);
    }

    private void validateAssignments(String msg) {
        for (int d = 0; d < D; d++) {
            doc_level_distr[d].validate(msg);
        }

        HashMap<SHLDANode, Integer> leafCustCounts = new HashMap<SHLDANode, Integer>();
        for (int d = 0; d < D; d++) {
            Integer count = leafCustCounts.get(c[d]);
            if (count == null) {
                leafCustCounts.put(c[d], 1);
            } else {
                leafCustCounts.put(c[d], count + 1);
            }
        }

        for (SHLDANode node : leafCustCounts.keySet()) {
            if (node.getNumCustomers() != leafCustCounts.get(node)) {
                throw new RuntimeException(msg + ". Numbers of customers mismach.");
            }
        }
    }

    private void validateModel(String msg) {
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);

        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();
            if (!isLeafNode(node)) {
                int numChildCusts = 0;
                for (SHLDANode child : node.getChildren()) {
                    stack.add(child);
                    numChildCusts += child.getNumCustomers();
                }

                if (numChildCusts != node.getNumCustomers()) {
                    throw new RuntimeException(msg + ". Numbers of customers mismatch."
                            + " " + numChildCusts + " vs. " + node.getNumCustomers());
                }
            }

            if (this.isLeafNode(node) && node.isEmpty()) {
                throw new RuntimeException(msg + ". Leaf node " + node.toString()
                        + " is empty");
            }
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model string
            StringBuilder modelStr = new StringBuilder();
            Stack<SHLDANode> stack = new Stack<SHLDANode>();
            stack.add(word_hier_root);
            while (!stack.isEmpty()) {
                SHLDANode node = stack.pop();

                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getIterationCreated()).append("\n");
                modelStr.append(node.getNumCustomers()).append("\n");
                modelStr.append(node.getRegressionParameter()).append("\n");
                modelStr.append(DirMult.output(node.getContent())).append("\n");

                for (SHLDANode child : node.getChildren()) {
                    stack.add(child);
                }
            }

            // assignment string
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(d)
                            .append(":").append(n)
                            .append("\t").append(z[d][n])
                            .append("\n");
                }
            }

            for (int d = 0; d < D; d++) {
                assignStr.append(d)
                        .append("\t").append(c[d].getPathString())
                        .append("\n");
            }

            // output to a compressed file
            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);

            ZipEntry modelEntry = new ZipEntry(filename + ModelFileExt);
            writer.putNextEntry(modelEntry);
            byte[] data = modelStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            ZipEntry assignEntry = new ZipEntry(filename + AssignmentFileExt);
            writer.putNextEntry(assignEntry);
            data = assignStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }
        try {
            inputModel(filepath);

            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Load the model from a compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputModel(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        // initialize
//        this.initializeModelStructure();
        this.emptyModels = new DirMult[L - 1];
        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l] = new DirMult(V, betas[l + 1], uniform);
        }

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + ModelFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));
        HashMap<String, SHLDANode> nodeMap = new HashMap<String, SHLDANode>();
        String line;
        while ((line = reader.readLine()) != null) {
            String pathStr = line;
            int iterCreated = Integer.parseInt(reader.readLine());
            int numCustomers = Integer.parseInt(reader.readLine());
            double regParam = Double.parseDouble(reader.readLine());
            DirMult dmm = DirMult.input(reader.readLine());

            // create node
            int lastColonIndex = pathStr.lastIndexOf(":");
            SHLDANode parent = null;
            if (lastColonIndex != -1) {
                parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
            }

            String[] pathIndices = pathStr.split(":");
            int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
            int nodeLevel = pathIndices.length - 1;
            SHLDANode node = new SHLDANode(iterCreated, nodeIndex,
                    nodeLevel, dmm, regParam, parent);

            node.changeNumCustomers(numCustomers);

            if (node.getLevel() == 0) {
                word_hier_root = node;
            }

            if (parent != null) {
                parent.addChild(node.getIndex(), node);
            }

            nodeMap.put(pathStr, node);
        }
        reader.close();

        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();
            if (!isLeafNode(node)) {
                node.fillInactiveChildIndices();
                for (SHLDANode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        validateModel("Loading model " + filename);
    }

    /**
     * Load the assignments of the training data from the compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        // initialize
        this.initializeDataStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + AssignmentFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                String[] sline = reader.readLine().split("\t");
                if (!sline[0].equals(d + ":" + n)) {
                    throw new RuntimeException("Mismatch");
                }
                z[d][n] = Integer.parseInt(sline[1]);
            }
        }

        for (int d = 0; d < D; d++) {
            String[] sline = reader.readLine().split("\t");
            if (Integer.parseInt(sline[0]) != d) {
                throw new RuntimeException("Mismatch");
            }
            String pathStr = sline[1];
            SHLDANode node = getNode(parseNodePath(pathStr));
            c[d] = node;
        }
        reader.close();

        validateAssignments("Load assignments from " + filename);
    }

    public double[] outputRegressionResults(
            double[] trueResponses,
            String predFilepath,
            String outputFile) throws Exception {
        BufferedReader reader = IOUtils.getBufferedReader(predFilepath);
        String line = reader.readLine();
        String[] modelNames = line.split("\t");
        int numModels = modelNames.length;

        double[][] predResponses = new double[numModels][trueResponses.length];

        int idx = 0;
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            for (int j = 0; j < numModels; j++) {
                predResponses[j][idx] = Double.parseDouble(sline[j]);
            }
            idx++;
        }
        reader.close();

        double[] finalPredResponses = new double[trueResponses.length];
        for (int d = 0; d < trueResponses.length; d++) {
            double sum = 0.0;
            for (int i = 0; i < numModels; i++) {
                sum += predResponses[i][d];
            }
            finalPredResponses[d] = sum / numModels;
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int i = 0; i < numModels; i++) {
            RegressionEvaluation eval = new RegressionEvaluation(
                    trueResponses, predResponses[i]);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();
            ArrayList<Measurement> measurements = eval.getMeasurements();

            if (i == 0) {
                writer.write("Model");
                for (Measurement measurement : measurements) {
                    writer.write("\t" + measurement.getName());
                }
                writer.write("\n");
            }
            writer.write(modelNames[i]);
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");
        }
        writer.close();

        return finalPredResponses;
    }

    public double[] regressNewDocumentsFlat(
            int[][] newWords, double[] newResponses,
            String predFile) throws Exception {
        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. "
                    + reportFolderpath);
        }
        String[] filenames = reportFolder.list();

        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        ArrayList<String> modelList = new ArrayList<String>();

        for (int i = filenames.length - 1; i >= 0; i--) {
            String filename = filenames[i];
            if (!filename.contains("zip")) {
                continue;
            }

            double[] predResponses = regressNewDocumentsFlat(
                    reportFolderpath + filename,
                    newWords,
                    reportFolderpath + IOUtils.removeExtension(filename) + ".diagnose");
            predResponsesList.add(predResponses);
            modelList.add(filename);

            if (verbose) { // for debugging only
                logln("state file: " + filename
                        + ". iter = " + iter);
                RegressionEvaluation eval = new RegressionEvaluation(
                        newResponses, predResponses);
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
                System.out.println();
            }
        }

        // write prediction values
        BufferedWriter writer = IOUtils.getBufferedWriter(predFile);
        for (String model : modelList) // header
        {
            writer.write(model + "\t");
        }
        writer.write("\n");

        for (int r = 0; r < newWords.length; r++) {
            for (int m = 0; m < predResponsesList.size(); m++) {
                writer.write(predResponsesList.get(m)[r] + "\t");
            }
            writer.write("\n");
        }
        writer.close();

        // average predicted response over different models
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }
        return finalPredResponses;
    }
    private ArrayList<SHLDANode> nodeList;
    private DirMult[] docTopics;
    private int[][] x;

    private double[] regressNewDocumentsFlat(
            String stateFile,
            int[][] newWords,
            String diagnoseFile) throws Exception {
        if (verbose) {
            logln("Perform regression using model from " + stateFile);
        }

        try {
            inputModel(stateFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // debug
        if (verbose) {
            logln(getCurrentState());
        }

        words = newWords;
        responses = null; // for evaluation
        D = words.length;
        numTokens = 0;
        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
        }

        // initialize
        nodeList = new ArrayList<SHLDANode>();
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();
            if (!node.isRoot() && node.getIterationCreated() < BURN_IN) {
                nodeList.add(node);
            }
            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        // initialize structure
        this.x = new int[D][];
        this.docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            this.x[d] = new int[words[d].length];
            this.docTopics[d] = new DirMult(nodeList.size(), 0.1 * nodeList.size(), 1.0 / nodeList.size());
        }

        // debug
        if (verbose) {
            logln("--- MAX ITER  " + MAX_ITER);
            logln("--- BURN IN  " + BURN_IN);
            logln("--- SAMPLE LAG  " + LAG);
            logln("--- V = " + V);
            logln("--- # documents = " + D); // number of groups
            logln("--- # tokens = " + numTokens);
            logln("--- # topics = " + nodeList.size());
            int[] topicSizes = new int[nodeList.size()];
            for (int k = 0; k < topicSizes.length; k++) {
                topicSizes[k] = nodeList.get(k).getContent().getCountSum();
            }
            logln("--- topic sizes: "
                    + " min = " + StatisticsUtils.min(topicSizes)
                    + ". max = " + StatisticsUtils.max(topicSizes)
                    + ". avg = " + MiscUtils.formatDouble(StatisticsUtils.mean(topicSizes))
                    + ". sum = " + StatisticsUtils.sum(topicSizes));

            int topicWordCount = 0;
            for (int k = 0; k < nodeList.size(); k++) {
                topicWordCount += nodeList.get(k).getContent().getCountSum();
            }
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }
            logln("--- # topicWords: " + nodeList.size() + ". " + topicWordCount);
            logln("--- # docTopics: " + docTopics.length + ". " + docTopicCount);
        }

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleX(d, n, !REMOVE, ADD);
            }
        }

        if (verbose) {
            logln("After initialization");
            int topicWordCount = 0;
            for (int k = 0; k < nodeList.size(); k++) {
                topicWordCount += nodeList.get(k).getContent().getCountSum();
            }
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }
            logln("--- # topicWords: " + nodeList.size() + ". " + topicWordCount);
            logln("--- # docTopics: " + docTopics.length + ". " + docTopicCount);
        }

        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleX(d, n, REMOVE, ADD);
                }
            }

            if (verbose && iter % LAG == 0) {
                logln("--- --- iter = " + iter + " / " + MAX_ITER);
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                double[] predResponses = new double[D];
                for (int d = 0; d < D; d++) {
                    double[] empDist = docTopics[d].getEmpiricalDistribution();
                    double predRes = 0.0;
                    for (int k = 0; k < empDist.length; k++) {
                        predRes += empDist[k] * nodeList.get(k).getRegressionParameter();
                    }
                    predResponses[d] = predRes;
                }

                predResponsesList.add(predResponses);
            }
        }

        if (verbose) {
            logln("After iterating");
            int topicWordCount = 0;
            for (int k = 0; k < nodeList.size(); k++) {
                topicWordCount += nodeList.get(k).getContent().getCountSum();
            }
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }
            logln("--- # topicWords: " + nodeList.size() + ". " + topicWordCount);
            logln("--- # docTopics: " + docTopics.length + ". " + docTopicCount);
        }

        // diagnose
        BufferedWriter writer = IOUtils.getBufferedWriter(diagnoseFile);
        for (int k = 0; k < nodeList.size(); k++) {
            String[] topWords = getTopWords(nodeList.get(k).getContent().getDistribution(), 15);
            writer.write("k = " + k + "\t" + nodeList.get(k).toString() + "\t");
            for (String topWord : topWords) {
                writer.write(topWord + " ");
            }
            writer.write("\n");
        }
        writer.write("\n");

        for (int d = 0; d < D; d++) {
            writer.write("d = " + d
                    + "\t" + MiscUtils.arrayToSVMLightString(docTopics[d].getCounts()).trim()
                    + "\n");
        }
        writer.close();

        // averaging prediction responses over time
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }
        return finalPredResponses;
    }

    private void sampleX(int d, int n, boolean remove, boolean add) {
        if (remove) {
            docTopics[d].decrement(x[d][n]);
            nodeList.get(x[d][n]).getContent().decrement(words[d][n]);
        }

        double[] logprobs = new double[nodeList.size()];
        for (int k = 0; k < logprobs.length; k++) {
            SHLDANode node = nodeList.get(k);
            double logPrior = Math.log(node.getContent().getCountSum() + docTopics[d].getCount(k));
//            double logPrior = docTopics[d].getLogLikelihood(k);
            double wordLlh = nodeList.get(k).getContent().getLogLikelihood(words[d][n]);
            logprobs[k] = logPrior + wordLlh;
        }
        int sampledX = SamplerUtils.logMaxRescaleSample(logprobs);
        x[d][n] = sampledX;

        if (add) {
            docTopics[d].increment(x[d][n]);
            nodeList.get(x[d][n]).getContent().increment(words[d][n]);
        }
    }

    /**
     * Perform regression on test documents in the same groups as in the
     * training data.
     *
     * @param newWords New documents
     * @param newResponses The true new responses. This is used to evaluate the
     * predicted values.
     */
    public double[] regressNewDocuments(
            int[][] newWords,
            double[] newResponses,
            String predFile) throws Exception {
        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist");
        }
        String[] filenames = reportFolder.list();

        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        ArrayList<String> modelList = new ArrayList<String>();

        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];
            if (!filename.contains("zip")) {
                continue;
            }

            // perform prediction
            double[] predResponses = regressNewDocuments(
                    reportFolderpath + filename,
                    newWords,
                    reportFolderpath + IOUtils.removeExtension(filename) + ".diagnose");
            predResponsesList.add(predResponses);
            modelList.add(filename);

            // output prediction result
            BufferedWriter writer = IOUtils.getBufferedWriter(reportFolderpath + IOUtils.removeExtension(filename) + "-pred.txt");
            writer.write(predResponses.length + "\n");
            for (int ii = 0; ii < predResponses.length; ii++) {
                writer.write(newResponses[ii] + "\t" + predResponses[ii] + "\n");
            }
            writer.close();

            if (verbose) { // for debugging only
                logln("state file: " + filename
                        + ". iter = " + iter);
                RegressionEvaluation eval = new RegressionEvaluation(
                        newResponses, predResponses);
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
                System.out.println();
            }
        }

        // write prediction values
        BufferedWriter writer = IOUtils.getBufferedWriter(predFile);
        for (String model : modelList) // header
        {
            writer.write(model + "\t");
        }
        writer.write("\n");

        for (int r = 0; r < newWords.length; r++) {
            for (int m = 0; m < predResponsesList.size(); m++) {
                writer.write(predResponsesList.get(m)[r] + "\t");
            }
            writer.write("\n");
        }
        writer.close();

        // average predicted response over different models
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }
        return finalPredResponses;
    }

    private double[] regressNewDocuments(
            String stateFile,
            int[][] newWords,
            String diagnoseFile) throws Exception {
        if (verbose) {
            logln("Perform regression using model from " + stateFile);
        }

        try {
            inputModel(stateFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        words = newWords;
        responses = null; // for evaluation
        D = words.length;

        // initialize structure
        initializeDataStructure();

        if (verbose) {
            logln("Loaded trained model: " + getCurrentState());
            int docLevelCount = 0;
            for (int d = 0; d < D; d++) {
                docLevelCount += doc_level_distr[d].getCountSum();
            }
            logln("--- doc level count = " + docLevelCount);
        }

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = rand.nextInt(L);
                doc_level_distr[d].increment(z[d][n]);
            }
            samplePathAssignments(d, !REMOVE, ADD, !OBSERVED, !EXTEND);
        }

        if (verbose) {
            logln("Initialized assignments: " + getCurrentState());
            int docLevelCount = 0;
            for (int d = 0; d < D; d++) {
                docLevelCount += doc_level_distr[d].getCountSum();
            }
            logln("--- doc level count = " + docLevelCount);
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                samplePathAssignments(d, REMOVE, ADD, !OBSERVED, !EXTEND);

                for (int n = 0; n < words[d].length; n++) {
                    sampleLevelAssignments(d, n, REMOVE, ADD, REMOVE, ADD, !OBSERVED);
                }
            }

            if (verbose && iter % LAG == 0) {
                logln("--- iter = " + iter + " / " + MAX_ITER);
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                double[] predResponses = getRegressionValues();
//                double[] predResponses = getRegressionValuesWithoutRoot();
                predResponsesList.add(predResponses);
            }
        }

        if (verbose) {
            logln("Final structure: " + getCurrentState() + "\n");
            int docLevelCount = 0;
            for (int d = 0; d < D; d++) {
                docLevelCount += doc_level_distr[d].getCountSum();
            }
            logln("--- doc level count = " + docLevelCount);
        }

        // debug
        diagnose(diagnoseFile);

        // averaging prediction responses over time
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }
        return finalPredResponses;
    }

//    private HashMap<String, ArrayList<Double>> readPredictionFile(String filepath) throws Exception{
//        HashMap<String, ArrayList<Double>> predictions = new HashMap<String, ArrayList<Double>>();
//        BufferedReader reader = IOUtils.getBufferedReader(filepath);
//        
//        String line = reader.readLine();
//        String[] filenames = line.split("\t");
//        
//        while((line = reader.readLine()) != null){
//            String[] sline = line.split("\t");
//            
//        }
//        
//        reader.close();
//    }
    public void experimentPredictionMethods(
            int[][] newWords,
            double[] newResponses,
            String predFile) throws Exception {


        BufferedReader reader = IOUtils.getBufferedReader(predFile);
        String line;
        reader.close();




        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist");
        }
        String[] filenames = reportFolder.list();

        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        ArrayList<String> modelList = new ArrayList<String>();

        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];
            if (!filename.contains("zip")) {
                continue;
            }
        }
    }

    public void diagnose(String filepath) throws Exception {
        StringBuilder str = new StringBuilder();
        for (int d = 0; d < D; d++) {
            SHLDANode[] path = getPathFromNode(c[d]);
            str.append(d).append(" -> ");
            for (int l = 0; l < L; l++) {
                str.append(path[l].getPathString())
                        .append(" (").append(MiscUtils.formatDouble(path[l].getRegressionParameter()))
                        .append(", ").append(doc_level_distr[d].getCounts()[l])
                        .append(")\t");
            }
            str.append("\n");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        writer.write(str.toString());
        writer.close();


//        private void getXMLTopWordsHierarchy(
//                SHLDANode node,
//                ArrayList<String> vocab,
//                int numWords,
//                StringBuilder str){
//            for(int i=0; i<node.getLevel(); i++)
//                str.append("\t");
//            str.append("<node id='").append(node.getPathString()).append("' ");
//            str.append("property='").append(node.getNumCustomers())
//                    .append("; ").append(node.getContent().getCountSum())
//                    .append("; ").append(MiscUtils.formatDouble(node.getRegressionParameter()))
//                    .append("' ");
//            str.append("topwords='");
//            String[] topWords = node.getTopWords(vocab, numWords);
//            for(String topWord : topWords)
//                str.append(topWord).append(" ");
//            str.append("'>\n");
//
//            for(S child : node.getChildren())
//                getXMLTopWordsHierarchy(child, vocab, numWords, str);
//
//            for(int i=0; i<node.getLevel(); i++)
//                str.append("\t");
//            str.append("</node>\n");
//        }
    }

    class SHLDANode extends TreeNode<SHLDANode, DirMult> {

        private final int born;
        private int numCustomers;
        private double regression;

        SHLDANode(int iter, int index, int level, DirMult content,
                double regParam, SHLDANode parent) {
            super(index, level, content, parent);
            this.born = iter;
            this.numCustomers = 0;
            this.regression = regParam;
        }

        public int getIterationCreated() {
            return this.born;
        }

        double getLogJointProbability(double gamma) {
            ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
            for (SHLDANode child : this.getChildren()) {
                numChildrenCusts.add(child.getNumCustomers());
            }
            return SamplerUtils.getAssignmentJointLogProbability(numChildrenCusts, gamma);
        }

        public double getRegressionParameter() {
            return this.regression;
        }

        public void setRegressionParameter(double reg) {
            this.regression = reg;
        }

        public int getNumCustomers() {
            return this.numCustomers;
        }

        public void decrementNumCustomers() {
            this.numCustomers--;
        }

        public void incrementNumCustomers() {
            this.numCustomers++;
        }

        public void changeNumCustomers(int delta) {
            this.numCustomers += delta;
        }

        public boolean isEmpty() {
            return this.numCustomers == 0;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[")
                    .append(getPathString())
                    .append(", ").append(born)
                    .append(", #ch = ").append(getNumChildren())
                    .append(", #c = ").append(getNumCustomers())
                    .append(", #o = ").append(getContent().getCountSum())
                    .append(", reg = ").append(MiscUtils.formatDouble(regression))
                    .append("]");
            return str.toString();
        }
    }
}
