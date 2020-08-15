/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Copyright (C) 2020 Charles Sturt University
 */
package weka.classifiers.trees;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveRange;

/**
 * <!-- globalinfo-start -->
 * Implementation of Smooth Private Forest, a differentially private decision
 * forest designed to minimize the number of queries required and the
 * sensitivity of those queries. Originally published in:<br>
 * <br>
 * Fletcher, S., & Islam, M. Z. (2017). Differentially private random decision
 * forests using smooth sensitivity. Expert Systems with Applications, 78,
 * 16-31.<br>
 * <br>
 *
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -N &lt;number of trees in forest&gt;
 *  Number of trees in forest.
 *  (default 10)
 * </pre>
 *
 * <pre>
 * -D &lt;number of display trees&gt;
 *  Number of trees to display in the output.
 *  (default 3)
 * </pre>
 *
 * <pre>
 * -T &lt;tree depth&gt;
 *  Tree depth option. If &lt;= 0 will be equal to the tree depth
 *  calculation from the paper
 *  (default -1)
 * </pre>
 * 
 * <pre>
 * -E &lt;epsilon&gt;
 *  The privacy budget (epsilon) for the exponential mechanism.
 *  (default 1.0)
 * </pre>
 *
 * <pre>
 * -P
 *  Whether or not to display flipped majorities, sensitivity information and
 *  true distributions in leaves.
 *  (default true)
 * </pre>
 *
 * <pre>
 * -C &lt;classname&gt;
 * Specify the full class name of the classifier to compare with Smooth Private
 * Forest.
 * </pre>
 *
 * <pre>
 * -S &lt;num&gt;
 *  Seed for random number generator.
 *  (default 1)
 * </pre>
 * <!-- options-end -->
 *
 *
 * @author Michael Furner (mfurner@csu.edu.au)
 * @version $Revision: 1.0$
 */
public class SmoothPrivateForest extends AbstractClassifier
        implements OptionHandler, Randomizable {

    /**
     * for serialization
     */
    static final long serialVersionUID = -21773316839123123L;

    /**
     * Number of trees in forest
     */
    protected int m_forestSize = 10;

    /**
     * Number of trees to display
     */
    protected int m_numDisplayTrees = 3;

    /**
     * Privacy budget for differential privacy (episilon in paper)
     */
    protected float m_privacyBudget = 1.0f;
    
    /**
     * Whether or not to display flipped majorities, sensitivity information and
     * true distributions in leaves. Set to False for outputting and sharing
     * differentially private forests.
     */
    protected boolean m_displayPrivateInformation = true;

    /**
     * Classifier to compare to. Will build before the SmoothPrivateForest.
     */
    protected Classifier m_comparisonClassifier = new weka.classifiers.trees.SysFor();

    /**
     * Random number generator seed.
     */
    protected int m_seed = 1;

    /**
     * Random object for use in algorithms
     */
    private Random m_random;

    /**
     * Array to store ensemble
     */
    private Classifier[] m_ensemble;

    /**
     * The dataset on which the classifier has been built.
     */
    private Instances m_ds;

    /**
     * Tree depth. If <= 0, will use the paper's balls-in-bins probablity 
     * process to automatically select tree depth.
     */
    protected int m_treeDepthOption = -1;
    
    /**
     * Calculated or option-specified tree depth.
     */
    private int m_treeDepth = -1;

    /**
     * The domains of the attributes before they are altered after splits.
     */
    private HashMap<Integer, LinkedList<Double>> m_urDomains;

    /**
     * Time taken for comparison classifier to build.
     */
    private long m_comparisonClassifierTime;

    /**
     * Time taken for comparison classifier to make predictions.
     */
    private long m_comparisonPredictionTime;

    /**
     * Accuracy of comparison classifier on training dataset.
     */
    private double m_comparisonClassificationAccuracy;

    /**
     * Time taken for Smooth Private Forest to build.
     */
    private long m_SPFTime;
    
    private boolean m_untrained = true;
    
    private String errorMessage = "";
    
    private int m_numberOfNumericalAttributes = -1;

    /**
     * Default classifier to setup SysFor.
     */
    public SmoothPrivateForest() {
        m_comparisonClassifier = new weka.classifiers.trees.SysFor();
    }
    
    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Sam Fletcher & Md Zahidul Islam");
        result.setValue(TechnicalInformation.Field.YEAR, "2017");
        result.setValue(TechnicalInformation.Field.TITLE, "Differentially private random decision forests using smooth sensitivity");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Expert Systems with Applications");
        result.setValue(TechnicalInformation.Field.PUBLISHER, "Elsevier");
        result.setValue(TechnicalInformation.Field.VOLUME, "78");
        result.setValue(TechnicalInformation.Field.PAGES, "16-31");
        result.setValue(TechnicalInformation.Field.URL, "http://dx.doi.org/10.1016/j.eswa.2017.01.034");

        return result;

    }

    /**
     * Returns capabilities of algorithm
     *
     * @return Weka capabilities of SysFor
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = super.getCapabilities();   // returns the object from weka.classifiers.Classifier

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.disable(Capabilities.Capability.MISSING_VALUES);
        result.disable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.disable(Capabilities.Capability.STRING_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.disable(Capabilities.Capability.NUMERIC_CLASS);
        result.disable(Capabilities.Capability.DATE_CLASS);
        result.disable(Capabilities.Capability.RELATIONAL_CLASS);
        result.disable(Capabilities.Capability.UNARY_CLASS);
        result.disable(Capabilities.Capability.NO_CLASS);
        result.disable(Capabilities.Capability.STRING_CLASS);
        return result;
    }

    /**
     * Return a description suitable for displaying in the
     * explorer/experimenter.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter
     */
    public String globalInfo() {
        return "Implementation of Smooth Private Forest, a differentially "
                + "private decision forest designed to minimize the number of"
                + " queries required and the sensitivity of those queries "
                + "which was published in: \n"
                + "Fletcher, S., & Islam, M. Z. (2017). Differentially private "
                + "random decision forests using smooth sensitivity. Expert "
                + "Systems with Applications, 78, 16-31"
                + "For more information, see:\n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * List the possible options from the superclass
     *
     * @return Options Enumerated
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>();
        newVector.addElement(new Option("\tNumber of trees in forest.\n"
                + "\t(default 10)", "N", 1, "-N"));
        newVector.addElement(new Option("\tNumber of trees to display.\n"
                + "\t(default 3)", "D", 1, "-D"));
        newVector.addElement(new Option("\tPrivacy budget for differential "
                + "privacy (episilon in paper)\n"
                + "\t(default 1.0)", "E", 1, "-E"));
        newVector.addElement(new Option("\tWhether or not to display flipped "
                + "majorities, sensitivity information and true distributions "
                + "in leaves. Set to False for outputting and sharing"
                + " differentially private forests.\n"
                + "\t(default true)",
                "P", 0, "-P"));
        newVector.addElement(new Option("\tSeed for random number generator.\n"
                + "\t(default 1)", "S", 1, "-S <num>"));
        newVector.addElement(new Option("\tFull name of base classifier.\n"
                + "\t(default: " + defaultComparisonClassifierString()
                + ((defaultComparisonClassifierOptions().length > 0)
                        ? " with options "
                        + Utils.joinOptions(defaultComparisonClassifierOptions()) + ")" : ")"),
                "C", 1, "-C"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    /**
     * Builds the Smooth Private Forest by splitting up the dataset into disjoint
     * subsets and using random trees. 
     * @param ds - dataset with which to build SPF
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances ds) throws Exception {

        m_random = new Random(m_seed);
        getCapabilities().testWithFail(ds);
        ds = new Instances(ds);
        
        ds.deleteWithMissingClass();

        //if this is a dataset with only the class attribute
        if (ds.numAttributes() == 1 || this.m_forestSize > ds.numInstances()) {
            
            if(ds.numAttributes() == 1) {
                errorMessage += "This is a dataset with only the class value and cannot be used.\n";
            }
            if(this.m_forestSize > ds.numInstances()) {
                errorMessage += "The forest size has been set higher than the number of instances in the dataset so the forest cannot be build.\n";
            }
            
            ZeroR zr = new ZeroR();
            zr.buildClassifier(ds);
            m_ensemble = new Classifier[1];
            m_ensemble[0] = zr;
            return;
        }

        m_ds = ds; //store so we can use information about the attributes

        //start off by sorting out the comparison classifier
        long startTimeComparison = System.currentTimeMillis();
        m_comparisonClassifier.buildClassifier(ds);
        long endTimeComparison = System.currentTimeMillis();
        m_comparisonClassifierTime = endTimeComparison - startTimeComparison;
        //do comparison classifier predictions
        int numCorrectlyClassified = 0;
        startTimeComparison = System.currentTimeMillis();
        for (int i = 0; i < m_ds.numInstances(); i++) {
            if(ds.get(i).classValue() == m_comparisonClassifier.classifyInstance(ds.get(i))) {
                numCorrectlyClassified++;
            }
        }
        endTimeComparison = System.currentTimeMillis();
        m_comparisonPredictionTime = endTimeComparison - startTimeComparison;
        m_comparisonClassificationAccuracy = (double) numCorrectlyClassified / m_ds.numInstances();
        m_comparisonClassificationAccuracy *= 100.0; //percentage

        //start timing the SPF and start building it.
        long startTimeSPF = System.currentTimeMillis();
        m_ensemble = new Classifier[m_forestSize];
        
        if(m_treeDepthOption <= 0) {
            m_treeDepth = calculateTreeDepth(m_ds);
        } //otherwise we will use the user selected value
        else {
            m_treeDepth = m_treeDepthOption;
        }

        //generate ur-domains
        m_urDomains = new HashMap<>();
        for (int j = 0; j < m_ds.numAttributes(); j++) {

            LinkedList<Double> temp = new LinkedList<>();
            if (m_ds.attribute(j).isNumeric()) {
                temp.add(m_ds.attributeStats(j).numericStats.min);
                temp.add(m_ds.attributeStats(j).numericStats.max);
            } else if (m_ds.attribute(j).isNominal()) {
                for (int v = 0; v < m_ds.attribute(j).numValues(); v++) {
                    temp.add((double) v);
                }
            } else {
                throw new UnsupportedAttributeTypeException("You can't use this kind of attribute.");
            }

            m_urDomains.put(j, temp);

        }

        int subsetSize = m_ds.numInstances() / m_forestSize;

        for (int t = 0; t < m_forestSize; t++) {

            int startIndex = t * subsetSize + 1;
            int endIndex = (t + 1) * subsetSize;

            //generate a subset using weka "RemoveRange" filter with selection inverted
            RemoveRange removeRangeFilter = new RemoveRange();
            removeRangeFilter.setInputFormat(m_ds);
            removeRangeFilter.setInvertSelection(true);
            removeRangeFilter.setInstancesIndices("" + startIndex + "-" + endIndex);
            Instances disjointSubset = Filter.useFilter(m_ds, removeRangeFilter);

            m_ensemble[t] = buildTree(disjointSubset, m_privacyBudget);

        }
        long endTimeSPF = System.currentTimeMillis();
        m_SPFTime = endTimeSPF - startTimeSPF;
        m_untrained = false;

    }

    /**
     * Sets up the root node which recursively builds the whole tree, and then
     * filters the records to the leaves and counts them with the exponential 
     * mechanism.
     * @param disjointSubset - a disjoint subset of the whole dataset
     * @param epsilon - the privacy budget
     * @return a random tree for the SPF
     * @throws Exception
     */
    public Classifier buildTree(Instances disjointSubset, double epsilon) throws Exception {

        int rootAttributeIndex;
        do {
            rootAttributeIndex = m_random.nextInt(m_ds.numAttributes());
        } while(rootAttributeIndex == m_ds.classIndex());

        SmoothPrivateTree tree = new SmoothPrivateTree(rootAttributeIndex, m_treeDepth);
        tree.filterTrainingDataAndCount(disjointSubset, epsilon);

        return (Classifier)tree;

    }

    /**
     * Calculates the required tree depth as per the paper. Only uses number 
     * of attributes of different types
     * @param ds - the dataset on which trees will be built
     * @return tree depth to use for SPF
     */
    public int calculateTreeDepth(Instances ds) {

        int retValue = -1;

        int m = 0;

        for (int i = 0; i < ds.numAttributes(); i++) {
            if (ds.attribute(i).isNumeric()) {
                m++;
            }
        }
        m_numberOfNumericalAttributes = m;

        if (m == 0) {
            retValue = (ds.numAttributes() - 1) / 2; //half the number of categorical atts
        } else {
            // Designed using balls-in-bins probability. See the paper for details.
            int depth = 0;
            double expectedEmpty = m;
            while (expectedEmpty > m / 2) {

                expectedEmpty = m * Math.pow((m - 1.0f) / m, depth);
                depth++;

            }
            retValue = (int) Math.floor(depth + ((ds.numAttributes() - 1 - m) / 2.0));

        }
        /* WARNING: The depth translates to an exponential increase in memory usage. 
         * Do not go above ~15 unless you have 50+ GB of RAM. */
        //System.out.println("Depth: " + Math.min(15, retValue));
        return Math.min(15, retValue);

    }

    /**
     * Returns seed for random number generator
     * @return seed for random number generator
     */
    @Override
    public int getSeed() {
        return m_seed;
    }

    /**
     * Sets seed for random number generator
     * @param i - seed for random number generator
     */
    @Override
    public void setSeed(int i) {
        m_seed = i;
    }
    
    /**
     * Weka tooltip
     * @return Weka tooltip
     */
    public String seedTipText() {
        return "Seed for random number generator.";
    }

    /**
     * Classifies record with majority voting on predictions from the SPF.
     * @param record - to classifiy
     * @return value index of predicted class
     * @throws Exception
     */
    @Override
    public double classifyInstance(Instance record) throws Exception {

         //if the forest hasn't been built 
        if (m_untrained) {
            return m_ensemble[0].classifyInstance(record);
        }
        
        int votes[] = new int[m_ds.numClasses()];
        for (int t = 0; t < m_ensemble.length; t++) {
            votes[((SmoothPrivateTree)m_ensemble[t]).classifyRecord(record)]++;
        }

        return (double) Utils.maxIndex(votes);

    }
    
    /**
     * Returns the vote distribution from the SPF
     * @param instance - record to classify
     * @return vote distribution for this instance
     * @throws Exception
     */
    @Override 
    public double[] distributionForInstance(Instance instance) throws java.lang.Exception {
        
         //if the forest hasn't been built 
        if (m_untrained) {
            return m_ensemble[0].distributionForInstance(instance);
        }
        
        double votes[] = new double[m_ds.numClasses()];
        for (int t = 0; t < m_ensemble.length; t++) {
            votes[((SmoothPrivateTree)m_ensemble[t]).classifyRecord(instance)]++;
        }
        
        for(int i = 0; i < votes.length; i++) {
            if(i == Utils.maxIndex(votes)) {
                votes[i] = 1.0;
            }
            else{
                votes[i] = 0.0;
            }
        }
        
        return votes;
        
    }

    /**
     * Outputs forest as String
     * @return forest as String
     */
    @Override
    public String toString() {

        //DecimalFormat df = new DecimalFormat("#0.00");
        String formatter = "#.";
        for(int i = 0; i < this.m_numDecimalPlaces; i++) {
            formatter += "#";
        }
        DecimalFormat df = new DecimalFormat(formatter);

        if (m_ensemble == null) {
            return "Forest not built!";
        }
        
        if(!"".equals(errorMessage)) {
            return errorMessage;
        }

        StringBuilder sb = new StringBuilder();

        for (int t = 0; t < m_numDisplayTrees; t++) {

            sb.append("Tree ").append(t + 1).append(": \n")
                    .append(((SmoothPrivateTree)m_ensemble[t]).toString(df))
                    .append("\n\n");

        }
        
        //Calculate the average sensitivity across all the leaves of all the trees
        double averageSensitivityAcrossAllTrees = 0;
        for (int t = 0; t < m_forestSize; t++) {
            averageSensitivityAcrossAllTrees += ((SmoothPrivateTree)m_ensemble[t]).averageSensitivity;
        }
        averageSensitivityAcrossAllTrees /= m_forestSize;

        if (m_numDisplayTrees < m_forestSize) {
            int tmp = m_forestSize - m_numDisplayTrees;
            sb.append(tmp)
                    .append(" trees hidden. Increase number of displayed trees to see more.\n");
        }

        sb.append("Smooth Private Forest of ").append(m_forestSize).append(" trees built in ")
                .append(m_SPFTime).append(" ms.")
                .append("\n").append("With privacy budget: ")
                .append(m_privacyBudget)
                .append("\n")
                .append("And an average sensitivity across all leaves of: ")
                .append(averageSensitivityAcrossAllTrees)
                .append("\n");
        
        if(m_treeDepthOption <= 0) {
            //TODO
            sb.append("Paper calculated tree depth via ")
              .append("d = argmin E[X|d] (d:X<")
              .append(m_numberOfNumericalAttributes)
              .append("/2), E[X|d] = ")
              .append(m_numberOfNumericalAttributes)
              .append("* (")
              .append(m_numberOfNumericalAttributes)
              .append(" - 1 /").append(m_numberOfNumericalAttributes)
              .append(" )^d")
              .append(" gave the result ")
              .append(m_treeDepth)
              .append("\n");
        }

        sb.append("Accuracy information will appear below. For comparison, the selected ")
                .append(m_comparisonClassifier.getClass().getName())
                .append(" was built in ")
                .append(m_comparisonClassifierTime)
                .append(" ms with a classification accuracy of ")
                .append(df.format(m_comparisonClassificationAccuracy))
                .append("% on the whole training dataset.\nThis comparison classifier's"
                        + " predictions were made in ")
                .append(m_comparisonPredictionTime)
                .append(" ms.");

        return sb.toString();

    }

    /**
     * Individual tree in the Smooth Private Forest
     */
    protected class SmoothPrivateTree extends AbstractClassifier implements Serializable {

        /**
         * The average sensitivity in the leaves.
         */
        protected double averageSensitivity = 0;
        
        /**
         * The root node of the tree
         */
        protected Node root;

        /**
         * Number of leaves in this tree
         */
        protected int numLeaves = 0;

        /**
         * Maximum depth for the tree (as per calculation from paper).
         */
        protected int maxDepth;

        /**
         * Number of times the exponential mechanism has got ,ajority class 
         * wrong on non-empty leaves
         */
        protected int numFlippedMajorities = -1;

        /**
         * Initialise and recursively build the SPT
         * @param rootAttribute - attribute that will be used as root 
         * @param treeDepth - max depth of the tree
         * @throws Exception
         */
        public SmoothPrivateTree(int rootAttribute, int treeDepth) throws Exception {

            this.maxDepth = treeDepth;

            root = new Node(null, null, rootAttribute, new LinkedList<>(), m_ds.numClasses());

            //if this is numerical
            if (m_ds.attribute(rootAttribute).isNumeric()) {

                LinkedList<Integer> availableAttrs = new LinkedList<>();
                for (int j = 0; j < m_ds.numAttributes(); j++) {
                    if (j != m_ds.classIndex()) {
                        availableAttrs.add(j);
                    }
                }

                double minVal = m_ds.attributeStats(rootAttribute).numericStats.min;
                double maxVal = m_ds.attributeStats(rootAttribute).numericStats.max;
                double splitPoint = m_random.nextDouble() * (maxVal - minVal) + minVal;

                HashMap<Integer, LinkedList<Double>> splitDomains = (HashMap< Integer, LinkedList<Double>>) m_urDomains.clone();

                //set up left domain
                HashMap< Integer, LinkedList<Double>> leftDomain = (HashMap< Integer, LinkedList<Double>>) splitDomains.clone();
                LinkedList<Double> leftList = new LinkedList<>();
                leftList.add(minVal);
                leftList.add(splitPoint);
                leftDomain.replace(rootAttribute, leftList);

                root.addChild(this.makeChildren(availableAttrs, root, 2, "<", splitPoint, leftDomain));
                HashMap< Integer, LinkedList<Double>> rightDomain = (HashMap< Integer, LinkedList<Double>>) splitDomains.clone();
                LinkedList<Double> rightList = new LinkedList<>();
                rightList.add(splitPoint);
                rightList.add(maxVal);
                rightDomain.replace(rootAttribute, rightList);
                root.addChild(this.makeChildren(availableAttrs, root, 2, ">=", splitPoint, rightDomain));

            } else if (m_ds.attribute(rootAttribute).isNominal()) { //if is categorical

                LinkedList<Integer> availableAttrs = new LinkedList<>();
                for (int j = 0; j < m_ds.numAttributes(); j++) {
                    if (j != rootAttribute && j != m_ds.classIndex()) {
                        availableAttrs.add(j);
                    }
                }

                for (int i = 0; i < m_ds.attribute(rootAttribute).numValues(); i++) {
                    //create a node (and its children) and add it to the current node as a child
                    Node childNode = this.makeChildren(availableAttrs, root, 2, "=", i, m_urDomains);
                    root.addChild(childNode);

                }

            } else {
                throw new UnsupportedAttributeTypeException("Unsupported Attribute Type");
            }

        }

        /**
         * Recursively make all the child nodes for the current node, until a 
         * termination condition is met
         * @param availableAttributes - attributes available for splits
         * @param parentNode - reference to parent node in tree
         * @param currentDepth - current depth of node
         * @param splitDirectionFromParent - "<", ">=" or =
         * @param parentNumericSplitPoint - if parent was split on numerical attr, the split point
         * @param domains - the available domains for the attributes
         * @return Node with all of its children attached
         */

        public Node makeChildren(LinkedList<Integer> availableAttributes, Node parentNode,
                int currentDepth, String splitDirectionFromParent, double parentNumericSplitPoint,
                HashMap<Integer, LinkedList<Double>> domains) {

            //termination conditions. leaf nodes don't count to the depth.
            if (availableAttributes.isEmpty() || currentDepth >= this.maxDepth + 1) {
                numLeaves++;
                return new Node(parentNode, splitDirectionFromParent, -1, null, m_ds.numClasses(), parentNumericSplitPoint);
            }

            //get a random attribute from those available
            int chosenAttribute = availableAttributes.get(m_random.nextInt(availableAttributes.size()));

            Node currentNode = new Node(parentNode, splitDirectionFromParent, chosenAttribute,
                    new LinkedList<Node>(), m_ds.numClasses(), parentNumericSplitPoint);

            if (m_ds.attribute(chosenAttribute).isNumeric()) { //numerical

                //set up the new domains so when we split we give the child nodes the right domains
                double minVal = m_ds.attributeStats(chosenAttribute).numericStats.min;
                double maxVal = m_ds.attributeStats(chosenAttribute).numericStats.max;
                double splitVal = m_random.nextDouble() * (maxVal - minVal) + minVal;

                //copy parent domain to edit it
                //set up left domain
                HashMap< Integer, LinkedList<Double>> leftDomain = (HashMap< Integer, LinkedList<Double>>) domains.clone();
                LinkedList<Double> leftList = new LinkedList<>();
                leftList.add(minVal);
                leftList.add(splitVal);
                leftDomain.replace(chosenAttribute, leftList);

                //add the left child to the current node with the correct domain
                currentNode.addChild(this.makeChildren(availableAttributes, currentNode, currentDepth + 1, "<", splitVal, leftDomain));

                HashMap< Integer, LinkedList<Double>> rightDomain = (HashMap< Integer, LinkedList<Double>>) domains.clone();
                LinkedList<Double> rightList = new LinkedList<>();
                rightList.add(splitVal);
                rightList.add(maxVal);
                rightDomain.replace(chosenAttribute, rightList);

                //add the left child to the current node with the correct domain
                currentNode.addChild(this.makeChildren(availableAttributes, currentNode, currentDepth + 1, ">=", splitVal, rightDomain));

            } else { //categorical

                LinkedList<Integer> availableAfterCategoricalSplit = (LinkedList<Integer>) availableAttributes.clone();
                availableAfterCategoricalSplit.removeFirstOccurrence(chosenAttribute);

                for (int i = 0; i < m_ds.attribute(chosenAttribute).numValues(); i++) {
                    //create a node (and its children) and add it to the current node as a child
                    Node childNode = this.makeChildren(availableAfterCategoricalSplit, currentNode, currentDepth + 1, "=", i, domains);
                    currentNode.addChild(childNode);

                }

            }

            return currentNode;

        }

        private void filterTrainingDataAndCount(Instances data, double epsilon) throws Exception {

            //filter each of the records from the root node to the leaf node they fit in
            for (int i = 0; i < data.numInstances(); i++) {
                filterRecord(data.get(i), root);
            }

            //set all noisy majorities
            numFlippedMajorities = this.setAllNoisyMajorities(epsilon, root);
            averageSensitivity /= numLeaves;

        }

        private void filterRecord(Instance record, Node node) throws Exception {

            if (node == null) {//debugging purposes
                throw new NullPointerException();
            }
            if (node.children == null || node.splittingAttribute == -1 || node.children.isEmpty()) {//i.e. this is a leaf

                node.incrementClassCount((int) record.classValue());
                return;

            }
            //if it's not a leaf filter it through to its correct child
            //pick the correct child for the node.
            Node child = null;
            if (m_ds.attribute(node.splittingAttribute).isNumeric()) {
                //find correct child for split value
                double thisRecVal = record.value(node.splittingAttribute);

                for (int i = 0; i < node.children.size(); i++) {

                    if (node.children.get(i).splitDirectionFromParent.contains("<")
                            && thisRecVal < node.children.get(i).parentNumericSplitPoint) {
                        child = node.children.get(i);
                        break;
                    }
                    if (node.children.get(i).splitDirectionFromParent.contains(">")
                            && thisRecVal >= node.children.get(i).parentNumericSplitPoint) {
                        child = node.children.get(i);
                        break;
                    }

                }

            } else if (m_ds.attribute(node.splittingAttribute).isNominal()) {
                //find correct child for categorical attr
                double thisRecVal = record.value(node.splittingAttribute);

                for (int i = 0; i < node.children.size(); i++) {
                    if (Double.compare(thisRecVal, node.children.get(i).parentNumericSplitPoint) == 0) {
                        child = node.children.get(i);
                        break;
                    }

                }

            } else {
                throw new UnsupportedAttributeTypeException();
            }

            filterRecord(record, child);

        }

        private int classifyRecord(Instance record) throws Exception {
            return classifyRecord(record, root);
        }

        private int classifyRecord(Instance record, Node node) throws Exception {

            if (node == null) {//debugging purposes
                throw new NullPointerException();
            }
            if (node.children == null || node.splittingAttribute == -1 || node.children.isEmpty()) {//i.e. this is a leaf

                return node.noisyMajority;

            }
            //if it's not a leaf filter it through to its correct child
            //pick the correct child for the node.
            Node child = null;
            if (m_ds.attribute(node.splittingAttribute).isNumeric()) {
                //find correct child for split value
                double thisRecVal = record.value(node.splittingAttribute);

                for (int i = 0; i < node.children.size(); i++) {

                    if (node.children.get(i).splitDirectionFromParent.contains("<")
                            && thisRecVal < node.children.get(i).parentNumericSplitPoint) {
                        child = node.children.get(i);
                        break;
                    }
                    if (node.children.get(i).splitDirectionFromParent.contains(">")
                            && thisRecVal >= node.children.get(i).parentNumericSplitPoint) {
                        child = node.children.get(i);
                        break;
                    }

                }

            } else if (m_ds.attribute(node.splittingAttribute).isNominal()) {
                //find correct child for categorical attr
                double thisRecVal = record.value(node.splittingAttribute);

                for (int i = 0; i < node.children.size(); i++) {
                    if (Double.compare(thisRecVal, node.children.get(i).parentNumericSplitPoint) == 0) {
                        child = node.children.get(i);
                        break;
                    }

                }

            } else {
                throw new UnsupportedAttributeTypeException();
            }

            return classifyRecord(record, child);

        }

        private int setAllNoisyMajorities(double epsilon, Node node) {

            //do the noisy majority calculation if its a leaf, otherwise filter down to the leaves
            if (node.children == null || node.splittingAttribute == -1 || node.children.isEmpty()) {
                double[] returnedArray = node.setNoisyMajority(epsilon);
                averageSensitivity += returnedArray[1];
                return (int)returnedArray[0];
            }

            int sum = 0;

            for (int i = 0; i < node.children.size(); i++) {
                sum += this.setAllNoisyMajorities(epsilon, node.children.get(i));
            }

            return sum; //sum of flipped majorities

        }

        /**
         * Output tree as string
         * @param df - decimal format for outputting the right number of decimal places
         * @return tree as string
         */
        public String toString(DecimalFormat df) {

            return root.toString(0, df);

        }

        /**
         * Unused
         * @param i
         * @throws Exception
         */
        @Override
        public void buildClassifier(Instances i) throws Exception {
            throw new UnsupportedOperationException("Not supported yet.");
        }

    }

    /**
     * A node in the Smooth Private Tree
     */
    protected class Node implements Serializable {

        /**
         * Node of this parent. Null at root
         */
        protected Node parentNode;

        /**
         * <, >= or = depending on split direction from parent.
         */
        protected String splitDirectionFromParent;

        /**
         * If parent's splitting attribute was numerical, the point at which the
         * dataset was split. If categorical, the index of the categorical value
         * from the parent.
         */
        protected double parentNumericSplitPoint;

        /**
         * Attribute this node splits on.
         */
        protected int splittingAttribute;

        /**
         * Children of this node, null if leaf.
         */
        protected LinkedList<Node> children;

        /**
         * Class counts in this node
         */
        protected int[] classCounts;

        /**
         * The majority class as calculated by the exponential mechanism.
         */
        protected int noisyMajority;

        /**
         * True if this node is empty
         */
        protected boolean empty;

        /**
         * Sensitivity of class counts
         */
        protected double sensitivity;

        /**
         * Sensitivity of sensitivity
         */
        protected double sensOfSens;

        /**
         * Noisy sensitivity
         */
        protected double noisySensitivity;

        /**
         * Constructor without split poit
         * @param parentNode
         * @param splitDirection
         * @param splittingAttribute
         * @param children
         * @param numClasses
         */
        public Node(Node parentNode, String splitDirection, int splittingAttribute,
                LinkedList<Node> children, int numClasses) {

            this.parentNode = parentNode;
            this.splitDirectionFromParent = splitDirection;
            this.splittingAttribute = splittingAttribute;
            this.children = children;
            this.parentNumericSplitPoint = Double.NaN;
            this.noisyMajority = Integer.MIN_VALUE;

            this.empty = false;
            this.sensitivity = -1;
            this.classCounts = new int[numClasses];

        }

        /**
         * Constructor
         * @param parentNode
         * @param splitDirection
         * @param splittingAttribute
         * @param children
         * @param numClasses
         * @param splitPoint
         */
        public Node(Node parentNode, String splitDirection, int splittingAttribute,
                LinkedList<Node> children, int numClasses, double splitPoint) {

            this(parentNode, splitDirection, splittingAttribute, children, numClasses);
            this.parentNumericSplitPoint = splitPoint;

        }

        /**
         * Adds node to children list
         * @param childNode - the child node
         */
        public void addChild(Node childNode) {
            this.children.add(childNode);
        }

        /**
         * Increase value for class by 1
         * @param classValue - index for a class
         */
        public void incrementClassCount(int classValue) {
            this.classCounts[classValue]++;
        }

        /**
         * Use exponential mechanism to estimate class
         * @param epsilon - privacy budget
         * @return flipped majorities and sensitivity
         */
        public double[] setNoisyMajority(double epsilon) {

            double[] returnArray = new double[2];
            
            //only run this once per leaf
            if (this.noisyMajority == Integer.MIN_VALUE && this.children == null) {

                //get highest and second-highest class counts
                int maxCC = -1;
                int secondCC = -1;
                for (int i = 0; i < classCounts.length; i++) {

                    if (classCounts[i] > maxCC) {
                        secondCC = maxCC;
                        maxCC = classCounts[i];
                    }

                    if (classCounts[i] > secondCC && classCounts[i] < maxCC) {
                        secondCC = classCounts[i];
                    }
                }

                //assign noisy majority
                for (int i = 0; i < classCounts.length; i++) {

                    if (classCounts[Utils.maxIndex(classCounts)] < 1) {

                        this.empty = true;
                        this.noisyMajority = m_random.nextInt(classCounts.length);
                        return returnArray;

                    } else {

                        int countDifference = maxCC - secondCC; //j in paper
                        this.sensitivity = Math.exp(-1.0 * countDifference * epsilon);
                        returnArray[1] = this.sensitivity;
                        this.sensOfSens = 1.0;
                        this.noisySensitivity = 1.0;

                        this.noisyMajority = this.expoMech(epsilon, this.sensitivity, this.classCounts);
                        if (this.noisyMajority != Utils.maxIndex(classCounts)) {
                            returnArray[0] = 1;
                            return returnArray;
                        } else {
                            returnArray[0] = 0; //this means the exponential mechanism got it right
                            return returnArray;
                        }
                    }

                }
            } //end the "run once" if statement

            return returnArray;

        }

        private int expoMech(double e, double s, int[] counts) {
            /* For this implementation of the Exponetial Mechanism, we use a piecewise linear scoring function,
               where the element with the maximum count has a score of 1, and all other elements have a score of 0. */

            double[] weighted = new double[counts.length];
            int maxCount = counts[Utils.maxIndex(counts)];
            double power = 0;
            double sum = 0;

            //get the weighted votes
            for (int i = 0; i < counts.length; i++) {

                if (counts[i] == maxCount) {

                    if (s < Double.parseDouble("1.0E-10")) {
                        power = 50; // e^50 is already astronomical. sizes beyond that dont matter
                    } else {
                        power = Double.min(50.0, (e * 1) / (2 * s));
                    }

                } else {
                    power = 0;
                }

                weighted[i] = Math.exp(power);
                sum += weighted[i];

            }

            HashMap<Integer, Double> dist = new HashMap<>();

            for (int i = 0; i < weighted.length; i++) {
                weighted[i] /= sum;
                dist.put(i, weighted[i]);
            }

            Distribution<Integer> tmp = new Distribution<>(dist);
            return tmp.sample();

        }

        /**
         * String representation of node
         * @param indent - the depth of this node
         * @param df - decimal format for displaying split points
         * @return String representation of node
         */
        public String toString(int indent, DecimalFormat df) {

            StringBuilder indentString = new StringBuilder();
            for (int i = 0; i < indent - 1; i++) {
                indentString.append("|   ");
            }

            StringBuilder sb = new StringBuilder();

            if (indent == 0) {
                //sb.append(this.children.get(0).splittingAttribute);
            } else {
                sb.append(indentString);
                sb.append(m_ds.attribute(this.parentNode.splittingAttribute).name());
                sb.append(" ");
                sb.append(splitDirectionFromParent);
                sb.append(" ");
                if ("=".equals(splitDirectionFromParent)) {
                    sb.append(m_ds.attribute(this.parentNode.splittingAttribute).value((int) parentNumericSplitPoint));
                } else {
                    sb.append(df.format(parentNumericSplitPoint));
                }
                //sb.append(" ---> ");
            }

            if (children == null || splittingAttribute == -1 || children.isEmpty()) {//i.e. this is a leaf

                sb.append(": ");
                sb.append(m_ds.classAttribute().value(this.noisyMajority));

                if (m_displayPrivateInformation) {

                    if (this.empty) {
                        sb.append(" *!* Empty leaf, class randomly assigned.");
                    } else {

                        //indicate if this is a flipped majority
                        if (this.noisyMajority != Utils.maxIndex(classCounts)) {
                            sb.append(" *!*");
                        }

                        //display true class distribution
                        sb.append(" {");
                        for (int i = 0; i < classCounts.length; i++) {
                            if (i != 0) {
                                sb.append(",");
                            }
                            sb.append(m_ds.classAttribute().value(i))
                                    .append(":")
                                    .append(classCounts[i]);
                        }

                        sb.append("} Sensitivity: ")
                                //.append(df.format(this.sensitivity));
                                .append(String.format("%."+getNumDecimalPlaces()+"e", this.sensitivity));
                    }

                }

            } else { //not a leaf, add the children

                //sb.append(m_ds.attribute(splittingAttribute).name()).append(": ");
                for (int i = 0; i < children.size(); i++) {
                    sb.append("\n");
                    sb.append(children.get(i).toString(indent + 1, df));
                }
            }

            return sb.toString();

        }

    }

    class Distribution<T> {

        ArrayList<Double> probs = new ArrayList<>();
        ArrayList<T> events = new ArrayList<>();
        double sumProb;

        Distribution(HashMap<T, Double> probs) {
            for (T event : probs.keySet()) {
                sumProb += probs.get(event);
                events.add(event);
                this.probs.add(probs.get(event));
            }
        }

        public T sample() {
            T value;
            double prob = m_random.nextDouble() * sumProb;
            int i;
            for (i = 0; prob > 0; i++) {
                prob -= probs.get(i);
            }
            return events.get(i - 1);
        }
    }

    /**
     * Get forest size
     * @return forest size
     */
    public int getForestSize() {
        return m_forestSize;
    }

    /**
     * Set forest size
     * @param m_forestSize - new forest size
     */
    public void setForestSize(int m_forestSize) {
        this.m_forestSize = m_forestSize;
    }

    /**
     * Weka tooltip
     * @return Weka tooltip
     */
    public String forestSizeTipText() {
        return "Size of the decision forest.";
    }

    /**
     * Get number of trees to display
     * @return number of trees to display
     */
    public int getNumDisplayTrees() {
        return m_numDisplayTrees;
    }

    /**
     * Set number of trees to display
     * @param m_numDisplayTrees - number of trees to display
     */
    public void setNumDisplayTrees(int m_numDisplayTrees) {
        this.m_numDisplayTrees = m_numDisplayTrees;
    }

    /**
     * Weka tooltip
     * @return Weka tooltip
     */
    public String numDisplayTreesTipText() {
        return "Amount of decision trees to display in the output.";
    }
    
    /**
     * Get number of trees to display
     * @return number of trees to display
     */
    public int getTreeDepthOption() {
        return m_treeDepthOption;
    }

    /**
     * Set the tree depth (<= 0 uses paper's calculation)
     * @param m_treeDepth - the tree depth (<= 0 uses paper's calculation)
     */
    public void setTreeDepthOption(int m_treeDepthOption) {
        this.m_treeDepthOption = m_treeDepthOption;
    }

    /**
     * Weka tooltip
     * @return Weka tooltip
     */
    public String treeDepthOptionTipText() {
        return "The tree depth (<= 0 uses paper's calculation).";
    }

    /**
     * Get the privacy budget (epsilon) for differential privacy
     * @return the privacy budget (epsilon) for differential privacy
     */
    public float getPrivacyBudget() {
        return m_privacyBudget;
    }

    /**
     * Set the privacy budget (epsilon) for differential privacy
     * @param m_privacyBudget - new privacy budget (epsilon) for differential privacy
     */
    public void setPrivacyBudget(float m_privacyBudget) {
        this.m_privacyBudget = m_privacyBudget;
    }

    /**
     * Weka tooltip
     * @return Weka tooltip
     */
    public String privacyBudgetTipText() {
        return "Privacy budget for differential privacy (epsilon in the paper).";
    }

    /**
     * Get whether or not to display actual class counts and sensitivity information.
     * In a real differential privacy setting, this would be set to false as the
     * algorithm would have no access to this information after the model is 
     * built.
     * @return
     */
    public boolean getDisplayPrivateInformation() {
        return m_displayPrivateInformation;
    }

    /**
     * Set whether or not to display actual class counts and sensitivity information
     * @param m_displayPrivateInformation - whether or not to display private info
     */
    public void setDisplayPrivateInformation(boolean m_displayPrivateInformation) {
        this.m_displayPrivateInformation = m_displayPrivateInformation;
    }

    /**
     * Weka tooltip
     * @return Weka tooltip
     */
    public String displayPrivateInformationTipText() {
        return "Whether to display actual class counts and information on the calculation process. "
                + "Disable to output actually differentially private trees that can be published ";
    }

    /**
     * Return the classifier built alongside SPF for comparison purposes
     * @return classifier built alongside SPF for comparison purposes
     */
    public Classifier getComparisonClassifier() {
        return m_comparisonClassifier;
    }

    /**
     * Set the classifier built alongside SPF for comparison purposes
     * @param newClassifier - the classifier to be built alongside SPF for comparison purposes
     */
    public void setComparisonClassifier(Classifier newClassifier) {
        m_comparisonClassifier = newClassifier;
    }

    /**
     * Weka tooltip
     * @return Weka tooltip
     */
    public String comparisonClassifierTipText() {
        return "Classifier to compare Smooth Private Forest's performance with.";
    }

    /**
     * Return the default comparison classifier options
     * @return empty String array.
     */
    protected String[] defaultComparisonClassifierOptions() {

        return new String[0];
    }

    /**
     * String describing default comparison classifier.
     * @return default comparison classifier sring
     */
    protected String defaultComparisonClassifierString() {

        return "weka.classifiers.trees.SysFor";
    }

    /**
     * Parses a given list of options.
     * 
     * <!-- options-start --> Valid options are:
     * <p/>
     *
     * <pre>
     * -N &lt;number of trees in forest&gt;
     *  Number of trees in forest.
     *  (default 10)
     * </pre>
     *
     * <pre>
     * -D &lt;number of display trees&gt;
     *  Number of trees to display in the output.
     *  (default 3)
     * </pre>
     * 
     * <pre>
     * -T &lt;tree depth&gt;
     *  Tree depth option. If &lt;= 0 will be equal to the tree depth
     *  calculation from the paper
     *  (default -1)
     * </pre>
     *
     * <pre>
     * -E &lt;epsilon&gt;
     *  The privacy budget (epsilon) for the exponential mechanism.
     *  (default 1.0)
     * </pre>
     *
     * <pre>
     * -P
     *  Whether or not to display flipped majorities, sensitivity information and
     *  true distributions in leaves.
     *  (default true)
     * </pre>
     *
     * <pre>
     * -C &lt;classname&gt;
     * Specify the full class name of the classifier to compare with Smooth Private
     * Forest.
     * </pre>
     *
     * <pre>
     * -S &lt;num&gt;
     *  Seed for random number generator.
     *  (default 1)
     * </pre>
     * <!-- options-end -->
     * 
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        String sNumberTrees = Utils.getOption('N', options);
        if (sNumberTrees.length() != 0) {
            setForestSize(Integer.parseInt(sNumberTrees));
        } else {
            setForestSize(10);
        }
        
        String sTreeDepthOption = Utils.getOption('T', options);
        if (sNumberTrees.length() != 0) {
            setTreeDepthOption(Integer.parseInt(sTreeDepthOption));
        } else {
            setTreeDepthOption(-1);
        }

        String sPrivacyBudget = Utils.getOption('E', options);
        if (sPrivacyBudget.length() != 0) {
            setPrivacyBudget(Float.parseFloat(sPrivacyBudget));
        } else {
            setPrivacyBudget(1.0f);
        }

        String sDisplay = Utils.getOption('D', options);
        if (sDisplay.length() != 0) {
            int displayTrees = Integer.parseInt(sDisplay);
            if (displayTrees > getForestSize()) {
                displayTrees = 3;
            }
            setNumDisplayTrees(displayTrees);
        } else {
            setNumDisplayTrees(3);
        }

        String sSeed = Utils.getOption('S', options);
        if (sSeed.length() != 0) {
            setSeed(Integer.parseInt(sSeed));
        } else {
            setSeed(1);
        }

        boolean bDisplayPrivateInformation = Utils.getFlag('P', options);
        setDisplayPrivateInformation(bDisplayPrivateInformation);

        String classifierName = Utils.getOption('C', options);
        if (classifierName.length() > 0) {
            setComparisonClassifier(AbstractClassifier.forName(classifierName, null));
            setComparisonClassifier(AbstractClassifier.forName(classifierName,
                    Utils.partitionOptions(options)));
        } else {
            setComparisonClassifier(AbstractClassifier.forName(defaultComparisonClassifierString(), null));
            String[] classifierOptions = Utils.partitionOptions(options);
            if (classifierOptions.length > 0) {
                setComparisonClassifier(AbstractClassifier.forName(defaultComparisonClassifierString(),
                        classifierOptions));
            } else {
                setComparisonClassifier(AbstractClassifier.forName(defaultComparisonClassifierString(),
                        defaultComparisonClassifierOptions()));
            }
        }

        super.setOptions(options);
    }

    
    /**
     * Gets the current settings of the classifier.
     *
     * @return the current setting of the classifier
     */
    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();

        result.add("-N");
        result.add("" + getForestSize());

        result.add("-D");
        result.add("" + getNumDisplayTrees());
        
        if(m_treeDepthOption <= 0) {
            result.add("-T");
            result.add("" + getTreeDepthOption());
        }

        result.add("-E");
        result.add("" + getPrivacyBudget());

        if (m_displayPrivateInformation) {
            result.add("-P");
        }

        result.add("-S");
        result.add("" + getSeed());

        result.add("-C");
        result.add(getComparisonClassifier().getClass().getName());
        
        String[] classifierOptions = ((OptionHandler)getComparisonClassifier()).getOptions();
        if (classifierOptions.length > 0) {
          result.add("--");
          Collections.addAll(result, classifierOptions);
        }

        return result.toArray(new String[result.size()]);
    }

    /**
     * Main method for testing this class.
     *
     * @param argv should contain the following arguments: -t training file [-T
     * test file] [-c class index]
     */
    public static void main(String[] argv) {       
        runClassifier(new SmoothPrivateForest(), argv);
    }

}
