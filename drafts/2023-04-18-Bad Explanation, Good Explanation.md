





# Notes


research in philosophy, psychology, and cognitive science of how people define, generate, select, evaluate, and present explanations, which argues that people employ certain cognitive biases and social expectations to the explanation process. This


The running hypothesis is that by building more transparent, interpretable, or explainable systems, users will be better equipped to understand and therefore trust the intelligent agents [129], [25], [65].

Researchers argue that people employ certain biases [82] and social expectations [72] when they generate and evaluate explanation, and I argue that such biases and expectations can improve human interactions with explanatory AI.

For example, de Graaf and Malle [34] argues that because people assign human-like traits to artificial agents, people will expect explanations using the same conceptual framework used to explain human behaviours.

For millennia, philosophers have asked the questions about what constitutes an explanation, what is the function of explanations, and what are their structure

Ultimately, it is a human–agent interaction problem

Explainable AI is just one problem inside human–agent interaction.

Everyday explanation: To focus on ‘everyday’ (or local) explanations as a tool and process for an agent, who we call the explainer, to explain decisions made by itself or another agent to a person, who we call the explainee. ‘Everyday’ explanations are the explanations of why particular facts (events, properties, decisions, etc.) occurred, rather than explanations of more general relationships, such as those seen in scientific explanation. We justify this focus based on the observation from AI literature that trust is lost when users cannot understand traces of observed behaviour or decisions [166], [129], rather than trying to understand and construct generalised theories. Despite this, everyday explanations also sometimes refer to generalised theories, as we will see later in Section 2, so scientific explanation is relevant, and some work from this area is surveyed in the paper.

Causality: While causality is important in explanation, this paper is not a survey on the vast work on causality. I review the major positions in this field insofar as they relate to the relationship with models of explanation.

Explanations are contrastive — they are sought in response to particular counterfactual cases, which are termed foils in this paper. That is, people do not ask why event P happened, but rather why event P happened instead of some event Q. This has important social and computational consequences for explainable AI. In Sections 2–4, models of how people provide contrastive explanations are reviewed.

Explanation are selected (in a biased manner) — people rarely, if ever, expect an explanation that consists of an actual and complete cause of an event. Humans are adept at selecting one or two causes from a sometimes infinite number of causes to be the explanation. However, this selection is influenced by certain cognitive biases. In Section 4, models of how people select explanations, including how this relates to contrast cases, are reviewed.


Probabilities probably don't matter — while truth and likelihood are important in explanation and probabilities really do matter, referring to probabilities or statistical relationships in explanation is not as effective as referring to causes. The most likely explanation is not always the best explanation for a person, and importantly, using statistical generalisations to explain why events occur is unsatisfying, unless accompanied by an underlying causal explanation for the generalisation itself

Explanations are social — they are a transfer of knowledge, presented as part of a conversation2 or interaction, and are thus presented relative to the explainer's beliefs about the explainee's beliefs. In Section 5, models of how people interact regarding explanations are reviewed

point: explanations are not just the presentation of associations and causes (causal attribution), they are contextual. While an event may have many causes, often the explainee cares only about a small subset (relevant to the context), the explainer selects a subset of this subset (based on several different criteria), and explainer and explainee may interact and argue about this explanation.


Idee: use context to select the most appropriate explanation of many explanations by using a large language model 

This example shows different types of questions being posed, and demonstrates that the explanatory agent will need to keep track of the state of the explanation; for example, by noting what it has already told the explainee, and may have to infer what the explainee has inferred themselves.



Viel über causality: die Autoren landen dann bei 

Bei explanation as a product 
Halpern and Pearl [58] define a model-based approach using structural causal models over two sets of variables: exogenous variables, whose values are determined by factors external to the model, and endogenous variables, whose values are determined by relationships with other (exogenous or endogenous) variables. Each endogenous variable has a function that defines its value from other variables. A context is an assignment of values to variables. Intuitively, a context represents a ‘possible world’ of the model. A model/context pair is called a situation. Given this structure, Halpern and


Folgende Tabelle such mit Pearl im Hintergrund:

Question	Reasoning	Description
What?	Associative	Reason about which unobserved events could have occurred given the observed events
How?	Interventionist	Simulate a change in the situation to see if the event still happens
Why?	Counterfactual	Simulating alternative causes to see whether the event still happen


Explanation as abductive reasoning:

Harman [62] labels this process “inference to the best explanation”. Thus, one can think of abductive reasoning as the following process: (1) observe some (presumably unexpected or surprising) events; (2) generate one or more hypothesis about these events; (3) judge the plausibility of the hypotheses; and (4) select the ‘best’ hypothesis as the explanation [78].

In this paper, abductive inference is not equated directly to explanation, because explanation also refers to the product and the social process; but abductive reasoning does fall into the category of cognitive process of explanation


Terminology destinction 

interpretability of a model as: the degree to which an observer can understand the cause of a decision. Explanation is thus one mode in which an observer may obtain understanding, but clearly, there are additional modes that one can adopt, such as making decisions that are inherently easier to understand or via introspection. I equate ‘interpretability’ with ‘explainability’.

A justification explains why a decision is good, but does not necessarily aim to give an explanation of the actual decision-making process [9].

Motivation for explanations 

Curiosity is one primary criterion that humans use, but other pragmatic reasons include examination

As such, this section is primarily concerned with why people ask for ‘everyday’ explanations of why specific events occur, rather than explanations for general scientific phenomena, although this work is still relevant in many cases.


Malle [112, Chapter 3], who gives perhaps the most complete discussion of everyday explanations in the context of explaining social action/interaction, argues that people ask for explanations for two reasons:
	•			1.To find meaning: to reconcile the contradictions or inconsistencies between elements of our knowledge structures.2.To manage social interaction: to create a shared meaning of something, and to change others' beliefs & impressions, their emotions, or to influence their actions.
Creating a shared meaning is important for explanation in AI. In many cases, an explanation provided by an intelligent agent will be precisely to do this — to create a shared understanding of the decision that was made between itself and a human observer, at least to some partial level.


Contrastive explanations 


The key insight is to recognise that one does not explain events per se, but that one explains why the puzzling event occurred in the target cases but not in some counterfactual contrast case.” — Hilton [72, p. 67].


Research shows that people do not explain the causes for an event per se, but explain the cause of an event relative to some other event that did not occur; that is, an explanation is always of the form “Why P rather than Q?”, in which P is the target event and Q is a counterfactual contrast case that did not occur, even if the Q is implicit in the question. This is called contrastive explanation.

idea of contrastive explanation should not be controversial if we accept the argument outlined in Section 2.2 that people ask for explanations about events or observations that they consider abnormal or unexpected from their own point of view [77], [73], [69]. In such cases, people expect to observe a particular event, but then observe another, with the observed event being the fact and the expected event being the foil.


Further, it can be beneficial in a more pragmatic way: if a person provides a foil, they are implicitly pointing towards the part of the model they do not understand



Types and levels of explanations 

The type of explanation provided to a question is dependent on the particular question asked; for example, asking why some event occurred is different to asking under what circumstances it could have occurred; that is, the actual vs. the hypothetical [102], [159]. However, for the purposes of answering why-questions, we will focus on a particular subset of philosophical work in this area.


Aristotle's Four Causes model, also known as the Modes of Explanation model, continues to be foundational for cause and explanation. Aristotle proposed an analytic scheme, classed into four different elements, that can be used to provide answers to why-questions [60]:
	•			1.Material: The substance or material of which something is made. For example, rubber is a material cause for a car tyre.2.Formal: The form or properties of something that make it what it is. For example, being round is a formal cause of a car tyre. These are sometimes referred to as categorical explanations.3.Efficient: The proximal mechanisms of the cause something to change. For example, a tyre manufacturer is an efficient cause for a car tyre. These are sometimes referred to as mechanistic explanations.4.Final: The end or goal of something. Moving a vehicle is an efficient cause of a car tyre. These are sometimes referred to as functional or teleological explanations.

Example:

single why-question can have explanations from any of these categories. For example, consider the question: “Why does this pen contain ink?”. A material explanation is based on the idea that the pen is made of a substance that prevents the ink from leaking out. A formal explanation is that it is a pen and pens contain ink. An efficient explanation is that there was a person who filled it with ink. A final explanation is that pens are for writing, and so require ink.


Kass and Leake [85] define a categorisation of explanations of anomalies into three types: (1) intentional; (2) material; and (3) social. The intentional and material categories correspond roughly to Aristotle's final and material categories, however, the social category does not correspond to any particular category in the models of Aristotle, Marr [119], or Dennett [35]. The social category refers to explanations about human behaviour that is not intentionally driven. Kass and Leake give the example of an increase in crime rate in a city, which, while due to intentional behaviour of individuals in that city, is not a phenomenon that can be said to be intentional. While individual crimes are committed with intent, it cannot be said that the individuals had the intent of increasing the crime rate — that is merely an effect of the behaviour of a group of individuals

Theorie for the structure of explanations:

https://ars.els-cdn.com/content/image/1-s2.0-S0004370218305988-gr003.gif

From these categories, Overton [139] provides a crisp definition of the structure of scientific explanations. He argues that explanations of phenomena at one level must be relative to and refer to at least one other level, and that explanations between two such levels must refer to all intermediate levels. For example, an arthropod (Entity) has eight legs (Data). Entities of this Kind are spiders, according to the Model of our Theory of arthropods. In this example, the explanation is constructed by appealing to the Model of insects, which, in turn, appeals to a particular Theory that underlies that Model. Fig. 4 shows the structure of a theory-data explanation, which is the most complex because it has the longest chain of relationships between any two levels.

https://ars.els-cdn.com/content/image/1-s2.0-S0004370218305988-gr004.gif

Social Dimension:

With respect to social explanation, Malle [112] argues that social explanation is best understood as consisting of three layers:
	•			1.Layer 1: A conceptual framework that outlines the assumptions people make about human behaviour and explanation.2.Layer 2: The psychological processes that are used to construct explanations.3.Layer 3: Language layer that specifies the type of linguistic structures people use in giving explanations.

XAI

Research indicates that people request only contrastive explanations, and that the cognitive burden of complete explanations is too great

could be argued that because models in AI operate at a level of abstraction that is considerably higher than real-world events, the causal chains are often smaller and less cognitively demanding, especially if they can be visualised. Even if one agrees with this, this argument misses a key point: it is not only the size of the causal chain that is important — people seem to be cognitively wired to process contrastive explanations, so one can argue that a layperson will find contrastive explanations more intuitive and more valuable.

It is a challenge because often a person may just ask “Why X?”, leaving their foil implicit

Lipton [102] states that the obvious solution is that a non-contrastive question “Why P?” can be interpreted by default to “Why P rather than not-P ?

inferred. As noted later in Section 4.6.3, concepts such as abnormality could be used to infer likely foils, but techniques for HCI, such as eye gaze [164] and gestures could be used to infer foils in some applications.

is an opportunity because, as Lipton [102] argues, explaining a contrastive question is often easier than giving a full causal attribution because one only needs to understand what is different between the two cases, so

Further, it can be beneficial in a more pragmatic way: if a person provides a foil, they are implicitly pointing towards the part of the model they do not understand

-> select explanations 

However, most existing work considers contrastive questions, but not contrastive explanations; that is, finding the differences between the two cases. Providing two complete explanations does not take advantage of contrastive questions. Section 4.4.1 shows that people use the difference between the fact and foil to focus explanations on the causes relevant to the question, which makes the explanations more relevant to the explainee.


### Social attribution of behaviour
 they showed that when explaining an individual's behaviour, the participants were able to produce explanations faster and more confidently than for groups, and that the traits that they assigned to individuals were judged to be less ‘extreme’ than those assigned to groups. In a second set of experiments, Susskind et al. showed that people expect more consistency in an individual's behaviour compared to that of a group. When presented with a behaviour that violated the impression that participants had formed of individuals or groups, the participants were more likely to attribute the individual's behaviour to causal mechanisms than the groups' behaviour.


### Cognitive selection: how do people select and evaluate explanations
None is more true than any of the others, but the particular context of the question makes some explanations more relevant than others.”

 When requested for an explanation, people typically do not have direct access to the causes, but infer them from observations and prior knowledge. Then, they select some of those causes as the explanation, based on the goal of the explanation. These two processes are known as causal connection (or causal inference), which is a processing of identifying the key causal connections to the fact; and explanation selection (or casual selection), which is the processing of selecting a subset of those causes to provide as an explanation.

## Causal connection: counterfactuals and mutability
To determine the causes of anything other than a trivial event, it is not possible for a person to simulate back through all possible events and evaluate their counterfactual cases. Instead, people apply heuristics to select just some events to mutate. However, this process is not arbitrary. This section looks at several biases used to assess the mutability of events; that is, the degree to which the event can be ‘undone’ to consider counterfactual cases. It shows that abnormality (including social abnormality), intention, time and controllability of events are key criteria.

### Temporality
 This supports the hypothesis that earlier events are considered less mutable than later events.
Miller and Gunasegaram [131] show that the temporality of events is important, in particular that people undo more recent events than more distal events. For instance, in one of their studies, they asked participants to play the role of a teacher selecting exam questions for a task. In one group, the teacher-first group, the participants were told that the students had not yet studied for their exam, while those in the another group, the teacher-second group, were told that the students had already studied for the exam. Those in the teacher-second group selected easier questions than those in the first, showing that participants perceived the degree of blame they would be given for hard questions depends on the temporal order of the tasks. This supports the hypothesis that earlier events are considered less mutable than later events.

### Controlability
When asked to undo events, participants overwhelmingly selected the intentional event as the one to undo, demonstrating that people mentally undo controllable events over uncontrollable events, irrelevant of the controllable events position in the sequence or whether the event was normal or abnormal


## Explanation selection
imilar to causal connection, people do not typically provide all causes for an event as an explanation. Instead, they select what they believe are the most relevant causes

## Abonormality (similiar to contrastive events)
Related to the idea of contrastive explanation, Hilton and Slugoski [77] propose the abnormal conditions model, based on observations from legal theorists Hart and Honoré [64]. Hilton and Slugoski argue that abnormal events play a key role in causal explanation. They argue that, while statistical notions of co-variance are not the only method employed in everyday explanations, the basic idea that people select unusual events to explain is valid. 

## Responsibility
The notions of responsibility and blame are relevant to causal selection, in that an event considered more responsible for an outcome is likely to be judged as a better explanation than other causes. In fact, it relates closely to necessity, as responsibility aims to place a measure of ‘degree of necessity’ of causes. An event that is fully responsible outcome for an event is a necessary cause.




# DJO: What are good explanations
The truth of likelihood of an explanation is considered an important criterion of a good explanation. However, Hilton [73] shows that the most likely or ‘true’ cause is not necessarily the best explanation. Truth conditions4 are a necessary but not sufficient criteria for the generation of explanations. While a true or likely cause is one attribute of a good explanation, tacitly implying that the most probable cause is always the best explanation is incorrect. As an example, consider again the explosion of the Challenger shuttle (Section 4.4.2), in which a faulty seal was argued to be a better explanation than oxygen in the atmosphere. This is despite the fact the ‘seal’ explanation is a likely but not known cause, while the ‘oxygen’ explanation is a known cause. Hilton argues that this is because the fact that there is oxygen in the atmosphere is presupposed; that is, the explainer assumes that the explainee already knows this.



# XAI

The notions of causal temporality and responsibility would be reasonably straightforward to capture in many models, however, if one can capture concepts such as abnormality, responsibility intentional, or controllability in models, this provides further opportunities.


Abnormality is a key criterion for explanation selection, and as such, the ability to identify abnormal events in causal chains could improve the explanations that can be supplied by an explanatory agent. While for some models, such as those used for probabilistic reasoning, identifying abnormal events would be straightforward, and for others, such as normative systems, they are ‘built in’, for other 

One important note to make is regarding abnormality and its application to “non-contrastive” why-questions. As noted in Section 2.6.2, questions of the form “Why P?” may have an implicit foil, and determining this can improve explanation. In some cases, normality could be used to mitigate this problem. That is, in the case of “Why P?”, we can interpret this as “Why P rather than the normal case Q?” [72]. For example, consider the application of assessing the risk of glaucoma [22]. Instead of asking why they were given a positive diagnosis rather than a negative diagnosis, the explanatory again could provide one or more default foils, which would be ‘stereotypical’ examples of people who were not diagnosed and whose symptoms were more regular with respect to the general population. Then, the question becomes why was the person diagnosed with glaucoma compared to these default stereotypical cases without glaucoma.
### Coherence & Simplicty & generality
Thagard [171] argues that coherence is a primary criterion for explanation. He proposes the Theory for Explanatory Coherence, which specifies seven principles of how explanations relate to prior belief. He argues that these principles are foundational principles that explanations must observe to be acceptable. They capture properties such as if some set of properties P explain some other property Q, then all properties in P must be coherent with Q; that is, people will be more likely to accept explanations if they are consistent with their prior beliefs. Further, he contends that all things being equal, simpler explanations — those that cite fewer causes — and more general explanations — those that explain more events —, are better explanations. The model has been demonstrated to align with how humans make judgements on explanations [151].

# Social Explanation

e verb to explain is a three-place predicate: Someone explains something to someone. Causal explanation takes the form of conversation and is thus subject to the rules of conversation.

For this, Grice [56] distinguishes four categories of maxims that would help to achieve the cooperative principle:
1.
Quality: Make sure that the information is of high quality — try to make your contribution one that is true. This contains two maxims: (a) do not say things that you believe to be false; and (b) do not say things for which you do not have sufficient evidence.

2.
Quantity: Provide the right quantity of information. This contains two maxims: (a) make your contribution as informative as is required; and (b) do not make it more informative than is required.

3.
Relation: Only provide information that is related to the conversation. This consists of a single maxim: (a) Be relevant. This maxim can be interpreted as a strategy for achieving the maxim of quantity.

4.
Manner: Relating to how one provides information, rather than what is provided. This consists of the ‘supermaxim’ of ‘Be perspicuous’, but according to Grice, is broken into ‘various’ maxims such as: (a) avoid obscurity of expression; (b) avoid ambiguity; (c) be brief (avoid unnecessary prolixity); and (d) be orderly.

Tetlock and Boettger [38], [169] investigated the effect of implicature with respect to the information presented, particularly its relevance, showing that when presented with additional, irrelevant information, people's implicatures are diluted


# Social aspects of XAI
I argue that, if we are to design and implement agents that can truly explain themselves, in many scenarios, the explanation will have to be interactive and adhere to maxims of communication, irrelevant of the media used. For example, what should an explanatory agent do if the explainee does not accept a selected explanation?


# Summary
In particular, we should take the four major findings noted in the introduction into account in our explainable AI models: (1) why-questions are contrastive; (2) explanations are selected (in a biased manner); (3) explanations are social; and (4) probabilities are not as important as causal links. I acknowledge that incorporating these ideas are not feasible for all applications, but in many cases, they have the potential to improve explanatory agents. I hope and expect that readers will also find other useful ideas from this survey.


[^1] https://reader.elsevier.com/reader/sd/pii/S0004370218305988?token=59289A7C33C7C75011C7FED1CAC019E65A9C206AF26EA03000B07A93C52317324C82511043CDDF3DC40BE920BA5E7938&originRegion=eu-west-1&originCreation=20230410130031









