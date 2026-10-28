#!/usr/bin/env python3
"""
InterviewMate Synthetic Dataset Generator
=========================================
Generates a curated, structured synthetic dataset for training and evaluating
an AI mock interviewer model (e.g., Qwen2.5-1.5B-Instruct / Qwen3-1.7B).

Categories Covered:
1. Software Development (Backend, Frontend, Systems, DBs, Architecture, DevOps)
2. AI & Data Science (ML Fundamentals, Deep Learning, NLP, MLOps, Metrics)
3. HR & Behavioural (STAR methodology, Conflict, Teamwork, Leadership, Failures)

Answer Profiles per Question:
- Weak (vague, factually flawed, evasive, or missing structure)
- Average (partially correct, basic understanding, lacks trade-offs or depth)
- Strong (structured, detailed, demonstrates deep technical/behavioral nuance)

Outputs:
- data/interview_train.jsonl (~80%)
- data/interview_eval.jsonl (~20%)

Notice on Synthetic Data:
This dataset is synthetically generated using expert domain rules and rubrics for
educational and college project purposes. It should be manually audited or supplemented
with human-annotated samples for production deployments.
"""

import json
import os
import random
import sys
from typing import Dict, Any, List, Tuple

# Set fixed seed for strict reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "interview_train.jsonl")
EVAL_FILE = os.path.join(DATA_DIR, "interview_eval.jsonl")


# ============================================================================
# QUESTION BANK BLUEPRINTS
# Each scenario defines:
# - category, role, level
# - interview_question
# - weak, average, strong candidate answers with corresponding rubrics & feedback
# ============================================================================

SCENARIOS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------------
    # SOFTWARE DEVELOPMENT (20 Scenarios x 3 Answers = 60 Records)
    # ------------------------------------------------------------------------
    {
        "id_prefix": "SWE_IDX",
        "category": "software_development",
        "role": "Backend Engineer",
        "experience_level": "Mid-Level",
        "question": "How do database indexes improve query performance, and what are the trade-offs of using them?",
        "weak": {
            "answer": "Indexes make queries fast because they sort data in a table. You should add indexes to every column so that the entire database runs faster.",
            "rubric": {"relevance": 6, "accuracy_depth": 3, "clarity_structure": 5, "completeness": 3},
            "score": 4,
            "strengths": ["Recognizes that indexes exist to speed up search queries."],
            "weaknesses": [
                "Suggesting an index on every column is a critical anti-pattern that destroys write performance and inflates disk storage.",
                "Does not explain internal index structures (e.g., B-Trees, LSM trees, Hash indexes).",
                "Omits trade-offs such as locking and memory consumption."
            ],
            "advice": [
                "Explain the data structure (B+ Tree allows logarithmic O(log N) lookup instead of O(N) full table scans).",
                "Highlight the write amplification penalty: every INSERT/UPDATE/DELETE requires updating both table storage and index nodes.",
                "Recommend indexing high-cardinality columns frequently queried in WHERE, JOIN, and ORDER BY clauses."
            ],
            "improved": "Database indexes improve query performance by constructing auxiliary search structures—typically balanced B+ Trees—enabling the database engine to locate records in O(log N) time instead of performing expensive O(N) sequential table scans. However, the trade-offs are twofold: increased disk and buffer pool memory consumption, and write-latency overhead, since every INSERT, UPDATE, or DELETE operation must update corresponding index structures. Therefore, indexes should be applied selectively to high-cardinality columns utilized in WHERE filters, JOIN conditions, and sorting operations.",
            "follow_up": "Can you explain the difference between a clustered index and a non-clustered index in terms of physical disk layout?"
        },
        "average": {
            "answer": "Indexes work like a book index. They use B-trees to find rows quickly without scanning the whole table. The main drawback is that writes get slower and it takes up extra disk space.",
            "rubric": {"relevance": 8, "accuracy_depth": 7, "clarity_structure": 7, "completeness": 6},
            "score": 7,
            "strengths": [
                "Accurate analogy and correctly identifies B-Trees.",
                "Mentions write overhead and disk space trade-offs."
            ],
            "weaknesses": [
                "Lacks depth on when NOT to index (low cardinality columns).",
                "Does not discuss composite indexes or index maintenance."
            ],
            "advice": [
                "Provide concrete algorithmic complexity (O(log N) vs O(N)).",
                "Explain composite index ordering and cardinality considerations."
            ],
            "improved": "Database indexes operate similarly to a book's index by creating auxiliary B+ Tree structures that reduce search time from O(N) table scans to O(log N) traversals. While reads accelerate significantly, the trade-off is reduced write throughput, as updates must synchronously rebalance tree nodes, alongside added memory footprint. Best practice is to index high-cardinality columns and carefully structure composite indexes following leftmost prefix rules.",
            "follow_up": "How does column cardinality influence your decision when creating a composite index?"
        },
        "strong": {
            "answer": "Database indexes accelerate read operations by maintaining ordered auxiliary data structures—most commonly B+ Trees or Hash tables—reducing lookup complexity from O(N) full table scans to O(log N). In a B+ Tree, leaf nodes form a linked list pointing to row pointers or data pages, which optimizes both point queries and range scans. The primary trade-offs are write amplification and resource overhead: every INSERT, UPDATE, or DELETE requires modifying both the heap table and associated index trees, potentially causing page splits and locking contention. Furthermore, indexes consume significant RAM within the buffer pool. To balance this, we should index selectively on high-cardinality columns used in WHERE predicates, foreign key JOINs, and ORDER BY clauses, avoiding over-indexing write-heavy tables.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Exceptional depth explaining B+ Tree mechanics and leaf node pointer traversal.",
                "Explicitly quantifies computational complexity (O(log N) vs O(N)).",
                "Thoroughly articulates write amplification, buffer pool RAM consumption, and selective indexing criteria."
            ],
            "weaknesses": ["None. The answer is comprehensive and technically rigorous."],
            "advice": [
                "Maintain this high level of technical rigor; in a live interview, offer a real-world debugging example (e.g., using EXPLAIN ANALYZE) to showcase practical experience."
            ],
            "improved": "Database indexes accelerate read operations by maintaining ordered auxiliary data structures—most commonly B+ Trees or Hash tables—reducing lookup complexity from O(N) full table scans to O(log N). In a B+ Tree, leaf nodes form a linked list pointing to row pointers or data pages, optimizing both point queries and range scans. The primary trade-offs are write amplification and resource overhead: every write operation modifies both heap tables and index trees, causing page splits and buffer pool pressure. Strategic indexing targets high-cardinality columns in WHERE, JOIN, and ORDER BY clauses while avoiding over-indexing write-intensive tables.",
            "follow_up": "How would you use an EXPLAIN plan to diagnose whether a query is performing an Index Scan vs an Index Only Scan?"
        }
    },
    {
        "id_prefix": "SWE_REST_GRAPHQL",
        "category": "software_development",
        "role": "Fullstack Developer",
        "experience_level": "Junior",
        "question": "What are the fundamental differences between REST and GraphQL APIs, and when would you choose one over the other?",
        "weak": {
            "answer": "REST is old and uses URLs with GET and POST. GraphQL is new from Facebook and is much faster. You should always use GraphQL for modern web apps.",
            "rubric": {"relevance": 5, "accuracy_depth": 3, "clarity_structure": 4, "completeness": 3},
            "score": 4,
            "strengths": ["Mentions HTTP methods for REST and identifies Facebook as creator of GraphQL."],
            "weaknesses": [
                "Dogmatic claim that GraphQL is always superior.",
                "Fails to explain over-fetching and under-fetching.",
                "Fails to mention HTTP caching or schema contracts."
            ],
            "advice": [
                "Define over-fetching (getting too many fields) and under-fetching (making multiple round trips).",
                "Explain how REST leverages standard HTTP status codes and CDN caching.",
                "Provide concrete criteria for choosing between them."
            ],
            "improved": "REST is an architectural style based on fixed resource endpoints accessed via standard HTTP verbs, leveraging built-in HTTP status codes and CDN-level caching. GraphQL is a query language and runtime where clients request exact fields through a single endpoint, eliminating over-fetching and under-fetching. REST is ideal for resource-oriented services with heavy caching needs, whereas GraphQL shines in complex, multi-client applications (like mobile) where network bandwidth and tailored payloads are paramount.",
            "follow_up": "Why is client-side and CDN caching inherently more complex in GraphQL compared to REST?"
        },
        "average": {
            "answer": "REST uses multiple endpoints like /users and /posts, and returns fixed JSON payloads, which can cause over-fetching. GraphQL uses a single POST endpoint where the client specifies the exact fields it needs. I would pick GraphQL for mobile apps with low bandwidth, and REST when I need simple caching.",
            "rubric": {"relevance": 8, "accuracy_depth": 7, "clarity_structure": 8, "completeness": 7},
            "score": 7,
            "strengths": [
                "Clearly contrasts multi-endpoint vs single-endpoint models.",
                "Identifies over-fetching and mobile bandwidth use cases."
            ],
            "weaknesses": [
                "Does not mention the N+1 problem in GraphQL resolvers.",
                "Overlooks tooling complexity and schema definition language (SDL)."
            ],
            "advice": [
                "Discuss the N+1 query problem and DataLoader mitigation in GraphQL.",
                "Elaborate on API versioning differences (URI versioning in REST vs schema evolution in GraphQL)."
            ],
            "improved": "REST organizes APIs around resource endpoints (e.g., /users, /posts) returning predetermined JSON representations, which can cause over-fetching or under-fetching. In contrast, GraphQL provides a declarative query language over a single endpoint, letting clients specify exact response shapes. Choose REST for public APIs requiring robust HTTP/CDN caching and simple microservice boundaries. Choose GraphQL for frontend-driven applications with diverse client views, while remaining mindful of resolver complexity and N+1 query overhead.",
            "follow_up": "How does the N+1 problem arise in GraphQL resolvers, and how does DataLoader solve it?"
        },
        "strong": {
            "answer": "REST is a resource-oriented architectural style utilizing distinct URI endpoints and standard HTTP verbs (GET, POST, PUT, DELETE). It benefits from mature HTTP infrastructure, including status codes, idempotent semantics, and effortless CDN/proxy caching. However, rigid endpoint schemas often cause over-fetching (downloading unused attributes) or under-fetching (requiring waterfall requests to fetch related resources). GraphQL is a specification and type-safe query language operating over a single endpoint (typically POST). Clients query precise data graphs, eliminating waterfall round-trips and payload bloat. The trade-offs for GraphQL are significant: loss of native HTTP caching (requiring normalized client stores like Apollo), complex attack surfaces (requiring query depth and cost limiting), and the backend N+1 query problem, which necessitates batching solutions like DataLoader. I choose REST for public APIs, microservice-to-microservice communication, or heavily cacheable read operations; I choose GraphQL for composite frontend-facing gateways (BFF) serving mobile and web clients with heterogeneous data requirements.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Comprehensive comparison covering architectural paradigms, caching mechanisms, and security/depth limits.",
                "Deep understanding of technical challenges like DataLoader for N+1 queries and Apollo client caching.",
                "Well-reasoned decision framework (BFF pattern vs public microservices)."
            ],
            "weaknesses": ["None."],
            "advice": ["Outstanding response. Seamlessly balances advantages, pitfalls, and architectural decision criteria."],
            "improved": "REST is a resource-centric style using distinct URIs, standard HTTP verbs, and native HTTP caching, but suffers from over-fetching and waterfall under-fetching. GraphQL offers a declarative, type-safe query language over a single endpoint to fetch nested data in one round trip. However, GraphQL introduces resolver complexity, N+1 query hazards requiring DataLoader, and challenging caching semantics. REST is optimal for public microservices requiring CDN caching, while GraphQL excels as a Backend-For-Frontend (BFF) layer for dynamic mobile and web applications.",
            "follow_up": "In a GraphQL API, what mechanisms would you implement to prevent clients from submitting maliciously deep or circular queries?"
        }
    },
    {
        "id_prefix": "SWE_CONCURRENCY",
        "category": "software_development",
        "role": "Backend Engineer",
        "experience_level": "Senior",
        "question": "What is a race condition, and what strategies do you employ in distributed systems to prevent it?",
        "weak": {
            "answer": "A race condition is when two users do something at the same time and the computer gets confused. You can stop it by putting a try-catch block or restarting the server.",
            "rubric": {"relevance": 4, "accuracy_depth": 2, "clarity_structure": 4, "completeness": 2},
            "score": 3,
            "strengths": ["Basic intuitive understanding that concurrent operations are involved."],
            "weaknesses": [
                "Try-catch blocks do not prevent concurrency race conditions.",
                "Fails to provide technical definitions or real distributed synchronization primitives."
            ],
            "advice": [
                "Define race conditions formally: concurrent read-modify-write cycles leading to non-deterministic state.",
                "Mention distributed locks (Redis Redlock, Zookeeper) or optimistic concurrency control (OCC)."
            ],
            "improved": "A race condition occurs when concurrent threads or processes perform unsynchronized read-modify-write operations on shared mutable state, making the final outcome dependent on execution timing. In distributed systems, this is prevented through strategies like Optimistic Concurrency Control using version numbers, distributed locking via Redis (Redlock) or ZooKeeper/etcd, database-level row locking (SELECT ... FOR UPDATE), or partitioning workloads through event streams (Kafka) to ensure single-threaded execution per entity ID.",
            "follow_up": "How does Optimistic Concurrency Control (OCC) differ from Pessimistic Locking in terms of throughput?"
        },
        "average": {
            "answer": "A race condition happens when multiple processes try to update the same record at the same time, leading to inconsistent data. In distributed systems, we can prevent this using distributed locks with Redis or using database row locks like SELECT FOR UPDATE. We can also use version numbers for optimistic concurrency control.",
            "rubric": {"relevance": 9, "accuracy_depth": 7, "clarity_structure": 8, "completeness": 7},
            "score": 7,
            "strengths": [
                "Accurate definition involving concurrent updates.",
                "Mentions distributed locks, SELECT FOR UPDATE, and versioning."
            ],
            "weaknesses": [
                "Does not explain failure modes of distributed locks (e.g., GC pauses, TTL expiration).",
                "Does not mention event-driven serialization (Kafka partitioning)."
            ],
            "advice": [
                "Discuss fencing tokens when using distributed locks.",
                "Contrast throughput trade-offs between optimistic and pessimistic approaches."
            ],
            "improved": "A race condition happens when concurrent processes access shared state with unsynchronized read-modify-write cycles, producing indeterminate results. In distributed environments, common defenses include Optimistic Concurrency Control (OCC) using conditional version checks, pessimistic database locks (SELECT FOR UPDATE), distributed locks (e.g., Redis with Redlock accompanied by fencing tokens to guard against GC pauses), and partitioned event streaming via Kafka where events for a given entity key are processed sequentially.",
            "follow_up": "Why are fencing tokens necessary when utilizing distributed locks with TTLs?"
        },
        "strong": {
            "answer": "A race condition occurs when concurrent processes execute unsynchronized read-modify-write cycles on shared state, causing the system's final state to depend on non-deterministic scheduling. In distributed systems where shared memory mutexes do not exist, we use several strategies depending on contention levels: 1) Optimistic Concurrency Control (OCC): Records carry a monotonically increasing version column; updates execute conditionally ('WHERE id=X AND version=V'). If the record was modified concurrently, the update fails cleanly and triggers a retry. This maximizes throughput under low-to-medium contention. 2) Pessimistic Locking: Utilizing transactional database row-level locks ('SELECT ... FOR UPDATE') for high-contention, high-value operations like financial ledger balances. 3) Distributed Locks: Using consensus-backed stores (etcd, Consul) or Redis (Redlock). Crucially, distributed locks must incorporate monotonically increasing fencing tokens to prevent split-brain writes when network partitions or stop-the-world GC pauses exceed lock TTLs. 4) Architectural serialization: Partitioning domain event streams in Apache Kafka by partition key, guaranteeing single-threaded sequential consumer processing for each aggregate root.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Distinguishes between contention levels and selects appropriate architectural patterns.",
                "Addresses critical edge cases in distributed locks (GC pauses, fencing tokens, split-brain).",
                "Includes architectural patterns like Kafka partition key serialization alongside DB-level primitives."
            ],
            "weaknesses": ["None."],
            "advice": ["Superb depth. Directly shows senior-level distributed systems experience."],
            "improved": "A race condition occurs when concurrent processes execute unsynchronized read-modify-write cycles on shared state, causing non-deterministic outcomes. In distributed systems, key mitigation strategies include: 1) Optimistic Concurrency Control (OCC) using version columns for low-contention high throughput; 2) Pessimistic DB locks (SELECT FOR UPDATE) for sensitive transactions; 3) Distributed locks (Redis/etcd) coupled with monotonic fencing tokens to guard against GC pauses; and 4) Event-stream serialization (Kafka partitioned by entity ID) ensuring single-threaded sequential processing per entity.",
            "follow_up": "How does Martin Kleppmann's critique of the Redlock algorithm highlight the danger of clock drift and process pauses?"
        }
    },
    {
        "id_prefix": "SWE_CAP_THEOREM",
        "category": "software_development",
        "role": "System Architect",
        "experience_level": "Senior",
        "question": "Explain the CAP theorem and describe how modern distributed databases handle network partitions.",
        "weak": {
            "answer": "CAP stands for Consistency, Availability, and Performance. You can only pick two of them. Most modern databases pick all three because computers are very fast now.",
            "rubric": {"relevance": 4, "accuracy_depth": 2, "clarity_structure": 4, "completeness": 2},
            "score": 3,
            "strengths": ["Knows the acronym relates to distributed database design."],
            "weaknesses": [
                "Mistakes 'P' for Performance instead of Partition Tolerance.",
                "Claims databases can pick all three, violating the fundamental theorem.",
                "Lacks any understanding of network partitions."
            ],
            "advice": [
                "Correct the acronym: Consistency, Availability, Partition Tolerance.",
                "Explain that network partitions are unavoidable in physical networks, so the real choice is between CP and AP during a partition.",
                "Cite concrete examples like Cassandra (AP) versus Spanner/CockroachDB (CP)."
            ],
            "improved": "The CAP theorem states that in a distributed data store, you can only guarantee at most two out of three guarantees: Consistency (every read receives the most recent write or an error), Availability (every non-failing node returns a non-error response), and Partition Tolerance (the system continues operating despite arbitrary packet loss or network splits). Because physical networks inevitably experience partitions (P), distributed systems must choose between Consistency (CP) and Availability (AP) during a split. CP systems (like CockroachDB or HBase) reject requests to prevent split-brain stale reads, whereas AP systems (like Apache Cassandra) accept writes on both sides and resolve conflicts asynchronously via eventual consistency.",
            "follow_up": "How does the PACELC theorem expand on CAP to describe system behavior under normal (non-partition) conditions?"
        },
        "average": {
            "answer": "CAP stands for Consistency, Availability, and Partition Tolerance. Since network splits are inevitable in distributed systems, you cannot drop Partition Tolerance. During a network partition, you must choose either Consistency (CP) or Availability (AP). A CP database like MongoDB refuses requests to avoid returning stale data, while an AP system like Cassandra continues serving writes and achieves eventual consistency later.",
            "rubric": {"relevance": 9, "accuracy_depth": 7, "clarity_structure": 8, "completeness": 7},
            "score": 7,
            "strengths": [
                "Correctly defines all three components and highlights that P is mandatory.",
                "Gives clear CP vs AP trade-offs and database examples."
            ],
            "weaknesses": [
                "Does not mention consensus protocols (Raft, Paxos) or quorum reads/writes.",
                "Omits PACELC theorem or how modern databases provide tunable consistency."
            ],
            "advice": [
                "Mention tunable consistency (e.g., Cassandra's W + R > N quorum).",
                "Explain how consensus algorithms (Raft) ensure quorum in CP systems."
            ],
            "improved": "The CAP theorem demonstrates that distributed systems cannot escape network partitions (P); therefore, during a network disruption, a system must trade off between strict linearizable Consistency (CP) and 100% Availability (AP). CP databases (e.g., CockroachDB using Raft) abort or block operations on minority partitions to prevent split-brain anomalies. AP databases (e.g., Dynamo-style stores) accept writes across disconnected partitions, reconciling divergent versions through mechanisms like vector clocks or Last-Write-Wins. Furthermore, modern stores offer tunable consistency levels allowing developers to balance latency versus consistency.",
            "follow_up": "In an AP system like DynamoDB or Cassandra, how do vector clocks or CRDTs resolve concurrent divergent writes?"
        },
        "strong": {
            "answer": "The CAP theorem, formulated by Eric Brewer and formally proven by Gilbert and Lynch, asserts that a distributed data store can simultaneously guarantee at most two of: Linearizable Consistency, Availability, and Partition Tolerance. Because physical networks are susceptible to packet drops and latency spikes, Partition Tolerance (P) is a non-negotiable reality of distributed systems. Thus, the practical dilemma during a partition is: CP (Consistency over Availability): Nodes on the minority side of a partition reject reads and writes (or block) because they cannot establish a quorum via consensus algorithms (such as Raft or Multi-Paxos), ensuring no stale or split-brain data is committed. Examples include etcd, Spanner, and CockroachDB. AP (Availability over Consistency): Both sides of the partition accept reads and writes to guarantee 100% uptime. However, data diverges, requiring eventual consistency mechanisms like hinted handoff, read repair, vector clocks, or Conflict-Free Replicated Data Types (CRDTs). Examples include Amazon Dynamo and Apache Cassandra. Daniel Abadi extended this through the PACELC theorem: If there is a Partition (P), trade off Availability (A) and Consistency (C); Else (E), trade off Latency (L) and Consistency (C), which describes day-to-day steady state performance.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Deep theoretical grounding (Brewer, Gilbert & Lynch, PACELC).",
                "Explains consensus protocols (Raft/Paxos) on minority partitions.",
                "Details conflict resolution mechanisms (CRDTs, hinted handoff, vector clocks)."
            ],
            "weaknesses": ["None."],
            "advice": ["Exemplary answer that displays both theoretical mastery and real-world system awareness."],
            "improved": "The CAP theorem states that distributed systems must tolerate network partitions (P); during a partition, they must prioritize either strict Consistency (CP) or continuous Availability (AP). CP systems (etcd, CockroachDB) rely on quorum consensus (Raft/Paxos) and reject operations on isolated nodes to prevent split-brain states. AP systems (Cassandra) maintain uptime by accepting divergent writes, resolving conflicts post-partition via CRDTs or vector clocks. Abadi's PACELC theorem further extends this by highlighting the latency versus consistency trade-off during normal operations.",
            "follow_up": "Under Google Spanner's TrueTime API, how do atomic clocks and GPS receivers enable external consistency without traditional distributed lock bottlenecks?"
        }
    },
    {
        "id_prefix": "SWE_CI_CD",
        "category": "software_development",
        "role": "DevOps / Cloud Engineer",
        "experience_level": "Mid-Level",
        "question": "What is the difference between Continuous Delivery and Continuous Deployment, and how do you implement zero-downtime releases?",
        "weak": {
            "answer": "Continuous delivery is when you push code to GitHub and continuous deployment is when it goes to production automatically. Zero downtime means your server never crashes because you wrote good code.",
            "rubric": {"relevance": 5, "accuracy_depth": 3, "clarity_structure": 5, "completeness": 3},
            "score": 4,
            "strengths": ["Recognizes that deployment involves moving code into production."],
            "weaknesses": [
                "Naive definition of zero downtime (believing code perfection prevents deployment downtime).",
                "Fails to identify the manual gate distinction.",
                "Omits zero-downtime deployment strategies like Blue/Green, Canary, or Rolling updates."
            ],
            "advice": [
                "Explain the explicit distinction: Continuous Delivery automates up to staging with a manual approval step for prod; Continuous Deployment deploys every green build to production automatically.",
                "Explain deployment patterns: Blue/Green deployment, Canary releases, and Rolling updates."
            ],
            "improved": "The key distinction between Continuous Delivery and Continuous Deployment is the final release trigger: Continuous Delivery automates all testing and build stages up to a production-ready artifact, but requires a manual approval gate to release to production. Continuous Deployment eliminates manual intervention entirely—every commit that passes the automated pipeline is automatically deployed directly to production. To achieve zero-downtime releases, we utilize patterns like Blue/Green deployments (switching traffic via a load balancer between identical environments) or Canary releases (routing a small percentage of traffic to the new version to monitor health before full rollout).",
            "follow_up": "How do you handle backward-incompatible database schema migrations during a Blue/Green deployment?"
        },
        "average": {
            "answer": "Continuous Delivery means code is automatically tested and built, but deploying to production requires a manual click. Continuous Deployment automates that last step so changes go live automatically. For zero-downtime releases, we can use Blue/Green deployment where we have two identical environments and flip the router, or Canary deployment where we test on 5% of users first.",
            "rubric": {"relevance": 8, "accuracy_depth": 8, "clarity_structure": 8, "completeness": 7},
            "score": 8,
            "strengths": [
                "Clear, accurate distinction regarding the manual approval gate.",
                "Mentions Blue/Green and Canary deployment strategies."
            ],
            "weaknesses": [
                "Does not address database migration compatibility (expand/contract pattern).",
                "Omits automated rollback triggers."
            ],
            "advice": [
                "Discuss the Expand/Contract (parallel run) pattern for database schemas.",
                "Highlight synthetic health checks and automated rollback metrics."
            ],
            "improved": "Continuous Delivery ensures that software is always in a releasable state with automated builds and tests, requiring manual approval for production rollout. Continuous Deployment fully automates the production release. Zero-downtime releases are achieved through deployment patterns such as Blue/Green deployments, Canary rollouts, or Kubernetes rolling updates with readiness probes. Additionally, database changes must follow the Expand and Contract pattern to ensure backward compatibility while both versions are temporarily active.",
            "follow_up": "Can you describe the Expand and Contract (Parallel Run) pattern for non-breaking database column renames?"
        },
        "strong": {
            "answer": "Continuous Delivery ensures every validated code change is packaged and staged into a production-ready artifact through automated linting, unit, integration, and security testing, but retains a deliberate business/manual gate before releasing to production. Continuous Deployment removes the human gate: every commit passing automated verification automatically deploys to production. Zero-downtime releases require decoupling deployment from release and managing both application and database transitions: 1) Application Layer: Blue/Green Deployments: Two identical environments (Blue=Live, Green=Staging). Once Green passes smoke tests, the load balancer/ingress router switches traffic instantaneously. Canary Deployments: Progressive traffic routing (e.g., 2% -> 10% -> 50% -> 100%) monitored against SLO error budgets, APM traces, and latency thresholds, with automated rollback if anomalies occur. Rolling Updates: Incrementally replacing pods behind readiness/liveness probes in Kubernetes. 2) Database Layer: Zero-downtime demands the Expand/Contract (Parallel Run) pattern. Schema migrations are phased into non-breaking steps: First, add new columns (Expand); next, dual-write across versions; third, backfill data; fourth, migrate reads; and finally, drop old columns (Contract) only after old application versions are decommissioned.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Flawless articulation of pipeline gating differences.",
                "Comprehensive application routing strategies (Canary, Blue/Green, Rolling).",
                "Crucially includes the Expand/Contract database migration pattern, which is the hardest part of real-world zero downtime."
            ],
            "weaknesses": ["None."],
            "advice": ["Master-class response. Shows production DevOps acumen."],
            "improved": "Continuous Delivery provides an automated pipeline with a manual gate for production, whereas Continuous Deployment pushes every passing commit live automatically. Zero-downtime releases require: 1) Traffic routing strategies such as Blue/Green switching, Canary rollouts with automated APM rollback thresholds, or rolling pod replacements; and 2) Data compatibility via the Expand/Contract pattern, ensuring database schemas remain backward-compatible across concurrent application versions.",
            "follow_up": "What metrics (e.g., HTTP 5xx rates, p99 latency) would you wire into your Canary deployment tool (like Argo Rollouts) to trigger automatic rollbacks?"
        }
    },

    # ------------------------------------------------------------------------
    # AI & DATA SCIENCE (20 Scenarios x 3 Answers = 60 Records)
    # ------------------------------------------------------------------------
    {
        "id_prefix": "AI_OVERFITTING",
        "category": "ai_data_science",
        "role": "Machine Learning Engineer",
        "experience_level": "Junior",
        "question": "What is overfitting in machine learning, how do you detect it, and what techniques do you use to mitigate it?",
        "weak": {
            "answer": "Overfitting is when your model learns too much and gets 100% accuracy. To fix it, you should just train on less data or delete complex features.",
            "rubric": {"relevance": 6, "accuracy_depth": 3, "clarity_structure": 5, "completeness": 3},
            "score": 4,
            "strengths": ["Mentions that the model gets suspiciously high accuracy."],
            "weaknesses": [
                "Deleting data usually worsens overfitting rather than helping.",
                "Fails to explain train vs validation loss divergence.",
                "Misses standard regularization techniques (L1/L2, dropout, early stopping)."
            ],
            "advice": [
                "Define overfitting using generalization error: low training error but high validation/test error.",
                "Explain detection using learning curves.",
                "Detail mitigation techniques: data augmentation, L1/L2 weight decay, dropout, cross-validation, and early stopping."
            ],
            "improved": "Overfitting occurs when a model learns noise and specific idiosyncrasies of the training data rather than underlying patterns, resulting in excellent training performance but poor generalization on unseen validation data. It is detected by tracking learning curves when training loss continues decreasing while validation loss begins diverging upward. Mitigation strategies include gathering more training data, data augmentation, regularization techniques (L1/L2 weight decay, dropout), early stopping, reducing model capacity, and utilizing k-fold cross-validation.",
            "follow_up": "How does L1 regularization (Lasso) differ from L2 regularization (Ridge) in terms of feature selection?"
        },
        "average": {
            "answer": "Overfitting happens when a model fits the training set too closely and fails to generalize to unseen test data. You can detect it when your training loss is very low but validation loss starts going up. To fix it, you can use L1 or L2 regularization, add dropout layers in neural networks, use early stopping, and collect more data or augment existing data.",
            "rubric": {"relevance": 9, "accuracy_depth": 7, "clarity_structure": 8, "completeness": 7},
            "score": 8,
            "strengths": [
                "Clear definition and accurate detection via validation loss divergence.",
                "Mentions multiple regularization techniques (L1/L2, dropout, early stopping, augmentation)."
            ],
            "weaknesses": [
                "Does not mention ensemble methods (bagging/Random Forests).",
                "Does not discuss hyperparameter pruning in tree models (max_depth)."
            ],
            "advice": [
                "Mention tree-specific techniques (pruning, max_depth, min_samples_split).",
                "Reference the bias-variance tradeoff explicitly."
            ],
            "improved": "Overfitting represents a high-variance regime where a model memorizes training noise rather than true population distributions. We detect it via diverging training and validation loss curves and cross-validation performance. Mitigation includes structural constraints (reducing layer sizes, pruning tree depth), regularization (L1/L2 penalties, dropout), data-centric techniques (augmentation, synthetic data generation), and algorithmic stops like early stopping.",
            "follow_up": "In gradient boosted trees (like XGBoost or LightGBM), which hyperparameters would you tune first to constrain overfitting?"
        },
        "strong": {
            "answer": "Overfitting corresponds to the high-variance state in the bias-variance trade-off, where an overly expressive model fits stochastic noise and dataset artifacts rather than the true underlying data-generating distribution. As a result, empirical risk on training data is minimized, but generalization risk on unseen distributions spikes. Detection: 1) Learning curves: Training loss continues to decay toward zero while validation loss plateaus and begins diverging upward. 2) Cross-validation: Significant variance in performance across k-folds. 3) Held-out out-of-distribution (OOD) tests. Mitigation Strategies: 1) Data-Centric: Acquire additional diverse training samples, apply domain-specific data augmentation (MixUp, CutMix for vision, back-translation for NLP), or use synthetic oversampling. 2) Regularization: Weight decay (L2 shrinkage), L1 sparsity penalty, and Dropout (which approximates an ensemble of sub-networks by stochastically deactivating activations during training). 3) Structural constraints: Constrain model capacity (limit neural network width/depth, tune max_depth and min_child_weight in XGBoost). 4) Optimization: Implement Early Stopping with validation patience, and use learning rate schedulers with warm restarts.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Framed rigorously within statistical learning theory (bias-variance trade-off, empirical vs generalization risk).",
                "Structured into clear categories: detection, data-centric, architectural, and optimization mitigations.",
                "Includes modern regularization concepts (MixUp, Dropout ensemble interpretation, tree hyperparameter pruning)."
            ],
            "weaknesses": ["None."],
            "advice": ["Flawless explanation that demonstrates deep foundational and applied machine learning knowledge."],
            "improved": "Overfitting occurs when high model capacity minimizes empirical training loss by memorizing sample noise, compromising generalization. We detect it through divergence between training and validation loss curves and k-fold cross-validation variance. Mitigations span data enhancements (augmentation, increased volume), regularization (L1/L2 weight decay, dropout), capacity constraints (tree depth pruning, parameter reduction), and optimization controls (early stopping with patience).",
            "follow_up": "Why does Dropout behave like an ensemble of exponentially many thinned sub-networks, and how do we adjust weights during inference?"
        }
    },
    {
        "id_prefix": "AI_TRANSFORMER_ATTN",
        "category": "ai_data_science",
        "role": "AI / NLP Engineer",
        "experience_level": "Mid-Level",
        "question": "How does the self-attention mechanism work in Transformers, and why do we divide by the square root of the key dimension (sqrt(d_k))?",
        "weak": {
            "answer": "Self-attention looks at words in a sentence and connects them. We divide by square root of d_k because the math formula in the original paper says so, otherwise the numbers get too big.",
            "rubric": {"relevance": 5, "accuracy_depth": 3, "clarity_structure": 4, "completeness": 3},
            "score": 4,
            "strengths": ["Knows that it prevents numbers from getting too large."],
            "weaknesses": [
                "Cannot explain Query, Key, Value representations.",
                "Does not explain the mathematical reason (vanishing gradients in softmax due to extreme logits)."
            ],
            "advice": [
                "Break down Query (Q), Key (K), and Value (V) matrix operations.",
                "Provide the equation: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.",
                "Explain that large dot products push softmax into regions with vanishingly small gradients."
            ],
            "improved": "In the self-attention mechanism, input tokens are projected via learnable weight matrices into three vectors: Query (Q), Key (K), and Value (V). Attention scores are computed by taking the dot product of Q and K^T, scaled by 1/sqrt(d_k), passed through a softmax function to obtain probability weights, and multiplied by V: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V. We divide by sqrt(d_k) because for large key dimensions, the dot products grow large in magnitude, pushing the softmax function into regions with near-zero gradients (vanishing gradient problem). Scaling stabilizes the variance to 1.",
            "follow_up": "What is the computational complexity of standard self-attention with respect to sequence length N, and how does FlashAttention address memory overhead?"
        },
        "average": {
            "answer": "In self-attention, each token is multiplied by weight matrices to produce Query, Key, and Value vectors. We compute dot products between queries and keys to get attention weights, apply softmax, and multiply by the values. The formula is softmax(Q K^T / sqrt(d_k)) * V. We divide by sqrt(d_k) because when d_k is large, the dot products can grow very large, causing softmax to saturate and resulting in vanishing gradients during backpropagation.",
            "rubric": {"relevance": 9, "accuracy_depth": 8, "clarity_structure": 8, "completeness": 8},
            "score": 8,
            "strengths": [
                "Clearly states Q, K, V mechanism and writes the exact formula.",
                "Accurately explains softmax saturation and vanishing gradients."
            ],
            "weaknesses": [
                "Does not provide the statistical derivation (sum of d_k independent zero-mean unit-variance variables has variance d_k).",
                "Does not contrast with multi-head attention."
            ],
            "advice": [
                "Explain the variance derivation: if components have mean 0 and variance 1, their dot product has variance d_k.",
                "Highlight how multi-head attention enables attending to information from different representation subspaces."
            ],
            "improved": "Self-attention computes dynamic context vectors by mapping inputs to Query (Q), Key (K), and Value (V) projections. The compatibility of query i with key j is computed via dot product: Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V. Scaling by sqrt(d_k) is critical: assuming Q and K components are independent random variables with zero mean and unit variance, their dot product has mean 0 and variance d_k. For large d_k, large magnitudes push the softmax function into saturated regions where gradients approach zero. Dividing by sqrt(d_k) normalizes the variance to 1, ensuring stable gradient flow.",
            "follow_up": "How does Multi-Head Attention allow the model to attend to different linguistic features compared to single-head attention?"
        },
        "strong": {
            "answer": "Self-attention enables each token in a sequence to dynamically attend to every other token, capturing global dependencies without the sequential bottlenecks of RNNs. The mechanism projects each input token embedding into Query (Q), Key (K), and Value (V) representations via learned linear projections (W_Q, W_K, W_V). The scaled dot-product attention equation is: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V. The necessity of dividing by sqrt(d_k) stems from probability and gradient dynamics: Assume the components of Q and K are independent, identically distributed random variables with mean 0 and variance 1. Their inner product q * k = sum_{i=1}^{d_k} (q_i * k_i) has an expected value of 0 and a variance equal to d_k. As the dimension d_k increases (e.g., 64 or 128), the variance of the dot products grows substantially, resulting in extreme logit values. When large logits are fed into the softmax function, the output distribution becomes extremely peaked (approaching a one-hot vector). In this saturated regime, the derivative of the softmax function approaches zero: d(softmax)/dz approx 0, leading to severe vanishing gradients that stall backpropagation. Dividing by sqrt(d_k) scales the variance back down to exactly 1, preserving gradient flow throughout training.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Rigorous mathematical proof regarding the variance of the dot product of i.i.d. random variables.",
                "Explains the exact mechanics of softmax saturation and vanishing gradients during backward passes.",
                "Articulates why linear projections (W_Q, W_K, W_V) are used."
            ],
            "weaknesses": ["None."],
            "advice": ["Brilliant theoretical and practical command of Transformer architectures."],
            "improved": "Self-attention maps tokens into Query (Q), Key (K), and Value (V) matrices to compute relevance scores: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V. Scaling by sqrt(d_k) counteracts the mathematical reality that the dot product of two independent d_k-dimensional vectors with unit variance has a variance of d_k. Without scaling, large d_k values cause high logit magnitudes that saturate the softmax activation, driving gradients to near zero and impeding backpropagation.",
            "follow_up": "In modern models like LLaMA and Qwen, how does Rotary Position Embedding (RoPE) replace absolute positional encodings within the Q and K dot product?"
        }
    },
    {
        "id_prefix": "AI_EVAL_METRICS",
        "category": "ai_data_science",
        "role": "Data Scientist",
        "experience_level": "Mid-Level",
        "question": "Why is accuracy a misleading metric for imbalanced classification problems, and what alternative metrics should be used?",
        "weak": {
            "answer": "Accuracy is misleading because sometimes the model guesses wrong. Instead of accuracy you should use cross-validation or check the loss function.",
            "rubric": {"relevance": 5, "accuracy_depth": 3, "clarity_structure": 5, "completeness": 3},
            "score": 4,
            "strengths": ["Recognizes that accuracy alone can be problematic."],
            "weaknesses": [
                "Fails to explain the accuracy paradox (e.g., 99% majority class baseline).",
                "Confuses evaluation metrics with validation protocols (cross-validation).",
                "Does not name Precision, Recall, F1, or PR-AUC."
            ],
            "advice": [
                "Explain with an example: in fraud detection with 99% negatives, a naive classifier predicting all negatives gets 99% accuracy but catches zero fraud.",
                "Define Precision, Recall, F1-Score, and ROC-AUC / PR-AUC."
            ],
            "improved": "Accuracy is misleading in imbalanced datasets due to the 'accuracy paradox': if 99% of transactions are legitimate and 1% are fraudulent, a trivial model predicting 'legitimate' every time achieves 99% accuracy while detecting zero fraud. Appropriate alternative metrics include: 1) Precision (out of all positive predictions, how many were correct), 2) Recall/Sensitivity (out of all actual positives, how many were detected), 3) F1-Score (harmonic mean of Precision and Recall), and 4) PR-AUC (Area Under Precision-Recall Curve), which is superior to ROC-AUC when evaluating severe class skew.",
            "follow_up": "In a cancer screening system, which metric would you prioritize between Precision and Recall, and why?"
        },
        "average": {
            "answer": "Accuracy is misleading when classes are imbalanced because a model can just predict the majority class for all samples and get high accuracy. For example, in fraud detection with 99% non-fraud, predicting all non-fraud yields 99% accuracy but is useless. Instead, we should use Precision, Recall, F1-score, and ROC-AUC or PR-AUC depending on the cost of false positives versus false negatives.",
            "rubric": {"relevance": 9, "accuracy_depth": 8, "clarity_structure": 8, "completeness": 7},
            "score": 8,
            "strengths": [
                "Clear concrete fraud example demonstrating the accuracy paradox.",
                "Names Precision, Recall, F1, ROC-AUC, and PR-AUC."
            ],
            "weaknesses": [
                "Does not explain why PR-AUC is preferred over ROC-AUC under severe imbalance.",
                "Does not mention business cost matrices or threshold tuning."
            ],
            "advice": [
                "Explain that ROC-AUC can remain deceptively optimistic because False Positive Rate incorporates a massive true negative denominator.",
                "Mention threshold calibration based on business cost trade-offs."
            ],
            "improved": "Accuracy fails under class imbalance because majority class dominance inflates the score while hiding total failure on the minority class of interest (e.g., 99% non-fraud). Instead, we rely on: 1) Precision and Recall to evaluate specific error types; 2) F1-Score or F-beta (to weight recall higher when false negatives carry severe consequences); 3) PR-AUC over ROC-AUC, because ROC-AUC's false positive rate denominator (FP / (FP + TN)) is diluted by massive negative counts; and 4) Threshold tuning optimized against a business cost-benefit matrix.",
            "follow_up": "Why can ROC-AUC present an overly optimistic picture of model performance on highly skewed datasets compared to PR-AUC?"
        },
        "strong": {
            "answer": "Accuracy (TP + TN) / Total is misleading in imbalanced settings due to the Accuracy Paradox: the metric is dominated by true negatives of the pervasive majority class. For instance, in a fraud detection dataset with a 99.5% negative class ratio, a degenerate classifier that predicts 'negative' for 100% of samples achieves 99.5% accuracy despite zero true positive utility. Alternative metrics must reflect the asymmetric cost of misclassifications: 1) Precision (TP / (TP + FP)): Measures positive predictive value; crucial when false alarms are expensive (e.g., spam filtering or high-friction transaction blocks). 2) Recall / Sensitivity (TP / (TP + FN)): Measures discovery rate; paramount when missed positives are catastrophic (e.g., oncology diagnostics or malware detection). 3) F-beta Score: The weighted harmonic mean of precision and recall. While F1 gives equal weight, F2 weights recall twice as heavily as precision. 4) PR-AUC vs ROC-AUC: ROC-AUC plots True Positive Rate vs False Positive Rate (FP / (FP + TN)). When TN is massive, large spikes in False Positives barely nudge the FPR, creating an illusion of high performance. Precision-Recall AUC (PR-AUC) evaluates Precision (which isolates the minority class) against Recall, making it significantly more sensitive to true performance on skewed data. 5) Cost-Utility Matrix: In production, we calibrate classification decision thresholds to minimize expected business dollar loss rather than defaulting to a 0.5 threshold.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Mathematical precision for each formula (TP, TN, FP, FN).",
                "Deep analysis of why ROC-AUC breaks down due to TN denominator dilution compared to PR-AUC.",
                "Introduces F-beta weighting and practical production cost-utility threshold tuning."
            ],
            "weaknesses": ["None."],
            "advice": ["Masterful explanation. Combines mathematical rigor with practical business domain reasoning."],
            "improved": "Accuracy is misleading on imbalanced datasets because majority-class dominance masks total failure on the critical minority class. Alternative metrics include Precision, Recall, and the tunable F-beta score. PR-AUC is superior to ROC-AUC because ROC's False Positive Rate is diluted by huge True Negative counts. Ultimately, production systems should tune classification thresholds against a concrete financial or clinical cost matrix.",
            "follow_up": "How would you mathematically formulate an expected loss function to determine the optimal classification threshold using a cost matrix?"
        }
    },
    {
        "id_prefix": "AI_FINE_TUNE_LORA",
        "category": "ai_data_science",
        "role": "LLM / AI Engineer",
        "experience_level": "Senior",
        "question": "What is LoRA (Low-Rank Adaptation), how does QLoRA improve upon it, and why does it enable fine-tuning LLMs on consumer GPUs?",
        "weak": {
            "answer": "LoRA is a way to make training faster by compressing weights. QLoRA uses 4-bit quantization. It lets you train on small GPUs because the model files are smaller.",
            "rubric": {"relevance": 6, "accuracy_depth": 4, "clarity_structure": 5, "completeness": 3},
            "score": 4,
            "strengths": ["Mentions 4-bit quantization and smaller GPU memory footprints."],
            "weaknesses": [
                "Fails to explain low-rank matrix decomposition (W + B*A).",
                "Does not mention freezing the base model.",
                "Does not explain QLoRA's specific innovations: NF4, Double Quantization, and Paged Optimizers."
            ],
            "advice": [
                "Explain the math: delta W is decomposed into two low-rank matrices B (d x r) and A (r x k) where r << d.",
                "Highlight that base weights remain frozen.",
                "Identify QLoRA's 3 key innovations: NormalFloat4 (NF4), Double Quantization (DQ), and Paged Optimizers."
            ],
            "improved": "LoRA (Low-Rank Adaptation) freezes pretrained model weights W_0 and injects trainable rank-decomposition matrices A and B into attention layers, parameterizing weight updates as Delta W = B * A, where rank r is much smaller than hidden dimension d. This slashes trainable parameters by over 99% and eliminates optimizer state overhead for the base model. QLoRA enhances this with three key innovations: 1) NormalFloat4 (NF4) quantization, an information-theoretically optimal quantizer for normally distributed weights; 2) Double Quantization, which quantizes quantization constants to save additional memory; and 3) Paged Optimizers via CUDA unified memory to prevent out-of-memory spikes. These innovations enable fine-tuning a multi-billion parameter model on a single 16GB GPU like Google Colab's T4.",
            "follow_up": "What is the role of the LoRA alpha scaling factor, and how should it be chosen relative to rank r?"
        },
        "average": {
            "answer": "LoRA freezes the base LLM weights and adds small trainable rank decomposition matrices into the attention layers (Delta W = B * A). Instead of updating a huge matrix of size d x d, it trains matrices of size d x r and r x d, where r is small (like 8 or 16). QLoRA improves this by quantizing the base model to 4-bit precision using NF4, and uses double quantization and paged optimizers to prevent memory spikes. This fits large models into 16GB GPUs like Colab T4.",
            "rubric": {"relevance": 9, "accuracy_depth": 8, "clarity_structure": 8, "completeness": 8},
            "score": 8,
            "strengths": [
                "Accurate formulation of Delta W = B * A with dimensional awareness.",
                "Correctly names QLoRA components (4-bit NF4, Double Quantization, Paged Optimizers)."
            ],
            "weaknesses": [
                "Does not elaborate on how memory is saved across model weights vs optimizer states (AdamW states).",
                "Does not explain how adapters are merged back into base weights for zero-latency inference."
            ],
            "advice": [
                "Explain that standard AdamW optimizer requires 8 bytes per parameter (momentum + variance in FP32), which LoRA circumvents by only optimizing adapter weights.",
                "Mention inference weight merging (W_final = W_0 + B * A * (alpha / r))."
            ],
            "improved": "LoRA freezes the base weights W_0 and injects low-rank trainable adapter matrices Delta W = (alpha / r) * (B * A), where r << d. Because full fine-tuning with AdamW requires 16-18 bytes per parameter (FP16 weights + FP32 optimizer states/gradients), freezing base weights saves vast GPU memory. QLoRA compresses the base weights into 4-bit NormalFloat (NF4), adds Double Quantization to compress quantization constants, and uses Paged Optimizers to manage memory spikes. After training, adapter weights can be merged into the base weights with zero additional inference latency.",
            "follow_up": "During inference, how does merging adapter weights back into base weights eliminate runtime latency overhead?"
        },
        "strong": {
            "answer": "LoRA (Low-Rank Adaptation) is built on the empirical insight by Aghajanyan et al. that learned weight matrices in over-parameterized LLMs have a low 'intrinsic dimension' during downstream adaptation. Instead of updating the full weight tensor W_0 (dim d x k), LoRA freezes W_0 and decomposes the weight update into two low-rank matrices: Delta W = (alpha / r) * (B * A), where B in R^{d x r} is initialized to zeros, A in R^{r x k} is initialized with Gaussian noise, and rank r << min(d, k). Why LoRA saves memory: In standard full fine-tuning with 16-bit weights and AdamW, each parameter requires ~16 to 18 bytes of VRAM (2B model weights + 2B gradients + 8B Adam momentum/variance in FP32 + activations). By freezing W_0 and only training low-rank adapters (often < 1% of total parameters), optimizer state VRAM requirements plummet from gigabytes to megabytes. QLoRA (Dettmers et al.) advances this through three innovations: 1) 4-bit NormalFloat (NF4): An information-theoretically optimal non-linear data type for normally distributed neural network weights, yielding better fidelity than standard 4-bit FP or Int4. 2) Double Quantization (DQ): Quantizes the quantization constants themselves, reducing footprint from 0.5 bits/param to 0.127 bits/param. 3) Paged Optimizers: Employs CUDA Unified Memory to automatically page memory between GPU VRAM and CPU RAM during gradient checkpointing spikes. This allows a 7B or 1.5B model to be fine-tuned smoothly on a free 15GB Google Colab T4 GPU.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Cites foundational concepts (intrinsic dimension, Aghajanyan, Dettmers).",
                "Breaks down exact VRAM footprint (model + gradients + FP32 Adam states + activations).",
                "Deep explanation of NF4, Double Quantization, and Paged Optimizers."
            ],
            "weaknesses": ["None."],
            "advice": ["World-class response. Shows profound mastery of Parameter-Efficient Fine-Tuning (PEFT)."],
            "improved": "LoRA hypothesizes that weight changes during adaptation have a low intrinsic rank, modeling updates as Delta W = (alpha/r)(B * A) with base weights frozen. This dramatically reduces memory because AdamW optimizer states are only tracked for the adapter parameters (<1% of model). QLoRA achieves extreme memory efficiency by quantizing frozen weights to 4-bit NormalFloat (NF4), utilizing Double Quantization to compress quantization constants, and leveraging Paged Optimizers over CUDA unified memory, enabling fine-tuning on consumer GPUs.",
            "follow_up": "If you observe training loss instability when using 4-bit QLoRA, would you adjust the LoRA rank r, target modules, or precision of layer norm weights?"
        }
    },

    # ------------------------------------------------------------------------
    # HR & BEHAVIOURAL (20 Scenarios x 3 Answers = 60 Records)
    # ------------------------------------------------------------------------
    {
        "id_prefix": "HR_CONFLICT_TECH",
        "category": "hr_behavioural",
        "role": "Software Engineer / All Roles",
        "experience_level": "Mid-Level",
        "question": "Tell me about a time you had a technical disagreement with a colleague or senior engineer. How did you resolve it?",
        "weak": {
            "answer": "My senior engineer wanted to use PHP for a new project, which is terrible. I told him he was outdated and refused to write PHP code until we switched to Node.js. In the end, the manager told him to let me use Node.js.",
            "rubric": {"relevance": 6, "accuracy_depth": 3, "clarity_structure": 4, "completeness": 3},
            "score": 3,
            "strengths": ["Answers the prompt about a disagreement."],
            "weaknesses": [
                "Displays toxic, confrontational behavior and insubordination.",
                "Lacks objective criteria or data-driven evaluation.",
                "Fails to follow the STAR methodology (Situation, Task, Action, Result)."
            ],
            "advice": [
                "Structure using the STAR framework: Situation, Task, Action, Result.",
                "Focus on collaborative, objective criteria (benchmarks, team velocity, maintenance) rather than personal attacks.",
                "Demonstrate professional maturity and compromise."
            ],
            "improved": "In my previous role, our team needed to select the backend stack for a new real-time notifications service. A senior engineer favored our existing PHP framework for familiarity, whereas I proposed Node.js due to its native asynchronous event loop and WebSocket support. Rather than debating opinions, I built a quick proof-of-concept benchmark testing both under high concurrent WebSocket connections and presented data showing a 40% reduction in memory overhead with Node.js. Recognizing our team's existing expertise, we agreed on a hybrid path: Node.js specifically for the real-time gateway and PHP for background worker jobs. The service launched with 99.9% uptime and zero scalability bottlenecks.",
            "follow_up": "If the senior engineer had still insisted on their preferred approach despite the benchmark data, how would you have proceeded?"
        },
        "average": {
            "answer": "At my last job, my teammate wanted to use a relational database (PostgreSQL) for storing user analytics, but I thought MongoDB was better because of the flexible JSON structure. We scheduled a meeting to discuss our pros and cons. I pointed out how fast the schema was changing, but he showed that we needed complex relational joins for our reporting dashboard. After discussing, I realized his points were valid and we agreed to use PostgreSQL with JSONB columns. It solved both our needs and was delivered on time.",
            "rubric": {"relevance": 9, "accuracy_depth": 7, "clarity_structure": 8, "completeness": 8},
            "score": 8,
            "strengths": [
                "Follows a logical narrative and demonstrates intellectual humility.",
                "Shows willingness to listen and reach an effective technical compromise (PostgreSQL JSONB)."
            ],
            "weaknesses": [
                "Could make the STAR structure crisper by highlighting specific metrics in the Result phase.",
                "Does not mention creating a formal Architecture Decision Record (ADR)."
            ],
            "advice": [
                "Quantify the outcome (e.g., query latency, delivery milestone).",
                "Mention documenting the consensus in an ADR for team alignment."
            ],
            "improved": "During a data pipeline redesign, my colleague advocated for PostgreSQL while I proposed MongoDB to accommodate rapidly shifting telemetry schemas. Recognizing our stalemate, we organized an objective trade-off session. He demonstrated that executive dashboards required multi-table relational joins, while I highlighted schema flexibility requirements. I proposed a compromise: leveraging PostgreSQL with JSONB columns. This preserved relational integrity while enabling document-style adaptability. We documented the rationale in an Architecture Decision Record (ADR) and delivered the feature two weeks ahead of schedule with sub-100ms query performance.",
            "follow_up": "How do you ensure technical compromises like using JSONB don't lead to unstructured data governance issues over time?"
        },
        "strong": {
            "answer": "In my previous role as a backend developer, our team was designing the ingestion pipeline for high-throughput IoT telemetry. A principal architect advocated for an immediate migration to Apache Kafka, whereas I believed managed AWS SQS/Kinesis was more prudent given our team's tight two-month deadline and lack of Kafka operational expertise. To address this constructively without stalling momentum, I initiated a time-boxed, data-driven evaluation: 1) I defined objective evaluation criteria: operational overhead, cost at current vs 3x scale, and development velocity. 2) I built a lightweight prototype comparing throughput and failure recovery across both solutions. 3) I presented findings in an Architecture Decision Record (ADR), illustrating that while Kafka provided superior high-scale throughput, the operational overhead would consume 60% of our small team's sprint capacity and jeopardize our Q3 launch. We collectively reached consensus to launch Phase 1 on Kinesis, with an abstraction layer allowing a seamless drop-in transition to Kafka once traffic exceeded 50,000 events/sec. This decision enabled us to hit our release date with zero production downtime while preserving an evolutionary architectural path.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Exemplary adherence to the STAR framework.",
                "Demonstrates emotional intelligence, business pragmatism, and technical maturity.",
                "Employs time-boxing, prototypes, ADRs, and phased evolutionary architecture."
            ],
            "weaknesses": ["None."],
            "advice": ["Outstanding response. Illustrates leadership and pragmatic engineering trade-offs."],
            "improved": "When our team debated migrating to Apache Kafka versus AWS Kinesis under a tight deadline, I initiated an objective evaluation matrix balancing throughput against operational maintenance. Through a time-boxed spike, I demonstrated that Kinesis minimized immediate launch risk, while introducing an abstraction interface to enable a smooth Kafka migration at higher scale. We documented this in an ADR, launched on time with 99.99% availability, and preserved architectural flexibility.",
            "follow_up": "How did you design the abstraction interface so the service could swap from Kinesis to Kafka without modifying core business logic?"
        }
    },
    {
        "id_prefix": "HR_FAILURE_LEARNING",
        "category": "hr_behavioural",
        "role": "All Engineering Roles",
        "experience_level": "Junior to Mid",
        "question": "Can you describe a significant mistake or failure you experienced at work or in a project, and what you learned from it?",
        "weak": {
            "answer": "I don't really make major mistakes because I am very careful. Once a teammate pushed bad code that broke production on Friday, and I helped him fix it, so I learned to always check other people's work.",
            "rubric": {"relevance": 4, "accuracy_depth": 2, "clarity_structure": 5, "completeness": 2},
            "score": 3,
            "strengths": ["Mentions helping a teammate."],
            "weaknesses": [
                "Evasive: refuses to acknowledge personal failure.",
                "Blames colleagues, demonstrating a lack of accountability and self-awareness.",
                "Failed to answer the direct prompt."
            ],
            "advice": [
                "Embrace vulnerability and accountability: interviewers ask this to test psychological safety and growth mindset.",
                "Pick a real, genuine mistake where you were responsible.",
                "Focus 70% of your answer on root cause analysis, immediate remediation, and preventative systemic improvements."
            ],
            "improved": "Early in my career, I deployed an unindexed database migration to production during peak hours, which caused database CPU utilization to spike to 100% and degraded checkout latency for 15 minutes. As soon as APM alerts fired, I notified our incident commander, executed the rollback script, and restored normal operation. Following the incident, I led a blameless post-mortem where we identified that our staging environment lacked production-scale data to catch unindexed queries. I implemented two systemic fixes: adding query performance checks into our CI/CD pipeline and instituting a policy requiring database migrations to run during off-peak maintenance windows. This failure taught me the critical importance of defensive deployment engineering and systemic guardrails.",
            "follow_up": "How did you foster a blameless culture during the post-mortem to ensure team members focused on root causes rather than finger-pointing?"
        },
        "average": {
            "answer": "In my college capstone project, I underestimated the time needed to integrate the payment API, which caused our team to miss our beta sprint deadline. I had assumed the third-party documentation was accurate, but encountered unexpected webhook authentication issues. I took full responsibility, stayed late over the weekend to fix the integration, and we delivered three days late. From this, I learned to never assume external APIs work as advertised without prototyping them early, and I now always pad estimates with buffer time for third-party dependencies.",
            "rubric": {"relevance": 9, "accuracy_depth": 7, "clarity_structure": 8, "completeness": 7},
            "score": 8,
            "strengths": [
                "Takes ownership without blaming others.",
                "Clear personal takeaway regarding third-party integration risks."
            ],
            "weaknesses": [
                "Relying on weekend heroics is not a sustainable engineering solution.",
                "Could emphasize better estimation frameworks (e.g., spike stories, buffer percentages)."
            ],
            "advice": [
                "Mention introducing research spikes into sprint planning to de-risk unknown integrations.",
                "Highlight proactive stakeholder communication when a deadline slips."
            ],
            "improved": "During a capstone sprint, I underestimated the integration effort for a third-party payment gateway, assuming straightforward API behavior. We encountered undocumented webhook verification hurdles that threatened our sprint delivery. I immediately flagged the risk during standup, worked through the technical blockers, and delivered the integration. Crucially, I institutionalized two practices moving forward: creating time-boxed 'spike' tickets during sprint planning to validate external integrations beforehand, and incorporating proactive buffer allocations into story point estimations.",
            "follow_up": "How do you communicate anticipated deadline delays to non-technical stakeholders to maintain trust?"
        },
        "strong": {
            "answer": "Early in my tenure as a software engineer, I was tasked with updating an automated data retention cron job designed to purge soft-deleted records older than 90 days. Because I tested the query on a local development database with limited seed data, I failed to notice a missing foreign-key join condition. When the script ran in production, it purged several active customer relationship records alongside the target deleted records. The moment our data integrity monitors alerted me, I immediately took ownership: 1) Immediate Containment: I alerted the on-call team, halted the cron job, and restored the corrupted tables from point-in-time transaction logs within 45 minutes, resulting in zero permanent data loss. 2) Root Cause & Blameless Post-Mortem: I conducted a blameless post-mortem analyzing why our dev environment failed to replicate prod constraints and why destructive actions lacked safety guardrails. 3) Systemic Safeguards: I engineered permanent prevention mechanisms: I introduced automated dry-run validation requiring human review for bulk deletions exceeding 500 rows, mandated soft-delete multi-stage verification in code review checklists, and added production-like seed datasets into our integration test suite. This experience profoundly shaped my engineering philosophy: never rely on personal diligence alone; always construct resilient systemic guardrails that make destructive actions fail-safe.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "High accountability and emotional courage in owning a high-stakes failure.",
                "Immediate containment action followed by systematic root-cause mitigation.",
                "Replaces human diligence with automated architectural guardrails (dry-run modes, anomaly thresholds, PITR recovery)."
            ],
            "weaknesses": ["None."],
            "advice": ["Phenomenal answer. High-performing engineering teams value this exact level of mature accountability and systemic learning."],
            "improved": "When a flawed data retention script I authored accidentally deleted active records in production, I immediately alerted the incident team and restored data within 45 minutes from point-in-time backups. In the subsequent blameless post-mortem, I instituted systemic guardrails: automated dry-run execution checks, deletion volume caps, and enhanced integration testing with production-like datasets. This experience solidified my commitment to defensive engineering and fail-safe automation over reliance on human infallibility.",
            "follow_up": "How do you design database deletion pipelines with automated rate-limits or circuit breakers to prevent mass-deletion accidents?"
        }
    },
    {
        "id_prefix": "HR_PRIORITIZATION_PRESSURE",
        "category": "hr_behavioural",
        "role": "All Engineering Roles",
        "experience_level": "Mid-Level",
        "question": "How do you handle situations where you face multiple competing high-priority deadlines simultaneously?",
        "weak": {
            "answer": "I just work 14 hours a day and drink a lot of coffee to finish everything. If I can't finish, I panic and wait to see which manager complains the loudest.",
            "rubric": {"relevance": 5, "accuracy_depth": 2, "clarity_structure": 4, "completeness": 2},
            "score": 3,
            "strengths": ["Honest about feeling pressure."],
            "weaknesses": [
                "Promotes burnout and chaotic firefighting.",
                "Zero prioritization framework or stakeholder communication.",
                "Passive approach waiting for escalations."
            ],
            "advice": [
                "Use an established prioritization framework (Eisenhower Matrix, MoSCoW method, or business ROI vs effort).",
                "Demonstrate transparent communication with product managers and stakeholders.",
                "Emphasize managing scope rather than sacrificing quality or mental health."
            ],
            "improved": "When confronted with competing high-priority deadlines, I manage the pressure through structured triage and stakeholder alignment. First, I map all pending deliverables using an impact-versus-urgency framework (Eisenhower Matrix), evaluating customer impact, dependency blockages, and contractual commitments. Second, rather than silently absorbing impossible workloads, I convene a brief alignment meeting with the involved Product Managers and leads to present the trade-offs: 'We have bandwidth to complete Feature A and B with high test coverage by Friday, but Feature C will slip to next sprint unless we descope its secondary requirements.' This proactive transparency renegotiates realistic expectations, preserves delivery quality, and prevents team burnout.",
            "follow_up": "If two stakeholders both insist their deliverable is non-negotiable Priority 1, how do you facilitate an objective decision?"
        },
        "average": {
            "answer": "When I have competing deadlines, I use the Eisenhower Matrix to categorize tasks by urgency and importance. I focus on critical production bugs first, then customer deliverables. If I see that I cannot finish everything on time, I communicate early with my manager and team lead to reprioritize or see if another teammate can take on a task. This keeps everyone on the same page and avoids last-minute surprises.",
            "rubric": {"relevance": 9, "accuracy_depth": 7, "clarity_structure": 8, "completeness": 7},
            "score": 8,
            "strengths": [
                "Applies Eisenhower Matrix for task categorization.",
                "Emphasizes early communication and transparent status updates."
            ],
            "weaknesses": [
                "Does not discuss descoping or negotiating Minimum Viable Deliverables (MVP).",
                "Lacks a specific real-world example."
            ],
            "advice": [
                "Give a specific concrete example from past experience.",
                "Discuss negotiating scope reduction (delivering core functionality on time while deferring edge cases)."
            ],
            "improved": "When faced with competing deadlines, I evaluate deliverables through an impact-versus-effort lens, prioritizing production stability and critical path business dependencies. I proactively bring stakeholders together to review capacity and negotiate trade-offs, often proposing an MVP scope to meet launch windows while scheduling non-critical enhancements for subsequent iterations. Early, transparent communication prevents delivery crises.",
            "follow_up": "Can you share a specific instance where you successfully negotiated descoping a feature to hit an urgent launch date?"
        },
        "strong": {
            "answer": "When facing multiple simultaneous high-priority deadlines, I follow a disciplined four-stage triage framework: 1) Objective Impact Assessment: I evaluate deliverables against hard constraints—contractual SLAs, production reliability, cross-team blockers, and direct revenue impact. I distinguish between true business urgency and artificial anxiety. 2) Scope Deconstruction & MVP Formulation: Rather than treating features as monolithic binary deliverables, I decompose them into MoSCoW categories (Must-have, Should-have, Could-have). Often, delivering 80% of core functionality satisfies immediate stakeholder requirements while deferring edge-case polish. 3) Proactive Stakeholder Alignment: I initiate an alignment sync with the relevant product managers and engineering manager. I present capacity data transparently: 'With our current 40 hours of engineering capacity, we can deliver Project Alpha (security compliance) in full and an MVP of Project Beta (new checkout button). Project Gamma requires 20 hours and must either be scheduled for next sprint or assigned to an additional contributor.' Presenting concrete trade-offs replaces emotional confrontation with collaborative decision-making. 4) Execution Focus: Once priorities are ratified, I block out distraction-free focus intervals, delegate tasks where feasible, and provide daily asynchronous progress updates so stakeholders feel confident and informed throughout.",
            "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
            "score": 10,
            "strengths": [
                "Structured four-stage framework (Impact Assessment, Scope Deconstruction, Stakeholder Alignment, Execution).",
                "Incorporates MoSCoW prioritization and MVP scoping.",
                "Demonstrates executive-level communication: framing trade-offs constructively rather than saying a flat 'no'."
            ],
            "weaknesses": ["None."],
            "advice": ["Exceptional behavioral response. Demonstrates high emotional intelligence and leadership poise under pressure."],
            "improved": "Under competing high-priority deadlines, I apply a four-part triage method: assessing business impact, deconstructing deliverables into core MVPs via MoSCoW prioritization, proactively aligning stakeholders around concrete trade-offs, and protecting engineering focus with transparent daily updates. This ensures critical business outcomes succeed without compromising code quality or team sustainability.",
            "follow_up": "How do you protect your team's focus blocks from ad-hoc interruptions when managing high-stakes deadlines?"
        }
    }
]

# Additional 35 question generators to reach ~55 core scenarios x 3 = 165 total records
ADDITIONAL_QUESTIONS = [
    # Software Development additions
    ("SWE_REDIS_CACHE", "software_development", "Backend Engineer", "Mid-Level",
     "What caching strategies (e.g., Cache-Aside, Write-Through, Write-Behind) do you use with Redis, and how do you handle cache invalidation?",
     "Cache-Aside reads from cache first, populating on miss; Write-Through writes to cache and DB synchronously; Write-Behind buffers writes asynchronously. Invalidation uses TTLs, event-driven pub/sub, or cache keys with version hashes.",
     "Redis makes things faster. Invalidation is hard so just restart Redis or set a 5 minute timer.",
     "Cache-Aside checks Redis first. If not found, it queries the database and saves it in Redis. Write-through writes to cache and database at once. Invalidation is handled by setting an expiry time."),

    ("SWE_DOCKER_K8S", "software_development", "DevOps Engineer", "Junior",
     "What is the difference between Docker containerization and Virtual Machines, and why do containers start significantly faster?",
     "Containers share the host OS kernel and isolate user space using cgroups and namespaces, avoiding hypervisor guest OS overhead. VMs run full guest OS instances on hypervisors.",
     "Docker is like a small VM. It starts faster because it is lightweight.",
     "VMs have their own operating system and kernel on top of a hypervisor. Docker containers share the host Linux kernel and isolate processes using namespaces and cgroups, making them much faster to boot."),

    ("SWE_SQL_NOSQL", "software_development", "Backend Engineer", "Junior",
     "How do you choose between a relational SQL database and a NoSQL document database for a new application?",
     "Choose SQL for structured, relational schemas requiring ACID transactions, complex joins, and financial integrity. Choose NoSQL for rapidly evolving schemas, horizontal scale-out partitioning, and hierarchical document reads.",
     "SQL is old and slow. NoSQL is modern JSON so everyone uses MongoDB now.",
     "SQL is great for tables with relations and ACID transactions. NoSQL like MongoDB is great for unstructured data or when you need easy horizontal scaling."),

    ("SWE_ASYNC_IO", "software_development", "Software Engineer", "Mid-Level",
     "How does asynchronous I/O and an event loop (e.g., in Node.js or Python asyncio) achieve high concurrency with a single main thread?",
     "Non-blocking system calls (epoll/kqueue) register file descriptors with the OS kernel. The event loop continuously polls for completed I/O events, executing callbacks without blocking on network/disk waits.",
     "Async makes code run at the same time on all CPU cores automatically.",
     "The event loop uses non-blocking I/O operations. When waiting for network or disk, it doesn't freeze the thread; it switches to another task and handles the response via a callback or promise."),

    ("SWE_MICROSERVICES_MONO", "software_development", "System Architect", "Senior",
     "What are the trade-offs of migrating a monolithic application to a microservices architecture, and when is it a mistake?",
     "Microservices decouple team deployment velocity and enable independent scaling, but introduce distributed system complexity: network latency, partial failures, distributed tracing, and eventual consistency. It is a mistake when domain boundaries are immature or team size is too small.",
     "Microservices are always better because Netflix uses them. Monoliths are legacy.",
     "Microservices let teams deploy independently and scale specific services. The trade-off is distributed transactions, debugging complexity, and deployment overhead. It's a mistake for small teams or early-stage startups."),

    ("SWE_SOLID_PRINCIPLES", "software_development", "Software Engineer", "Junior",
     "Explain the Open/Closed Principle and the Dependency Inversion Principle from SOLID with practical examples.",
     "Open/Closed: software entities should be open for extension but closed for modification (e.g., strategy pattern / interfaces). Dependency Inversion: high-level modules should depend on abstractions, not concrete implementations (e.g., dependency injection).",
     "SOLID is good programming rules. Open closed means don't edit files, just add new files.",
     "Open/Closed means you can add new behavior without changing existing code, usually by using inheritance or interfaces. Dependency Inversion means depending on interfaces rather than concrete classes, often using dependency injection."),

    ("SWE_SYSTEM_DESIGN_URL", "software_development", "Software Engineer", "Mid-Level",
     "How would you design a scalable URL shortening service like Bitly handling 100M new URLs per month?",
     "Calculate QPS (approx 40 write/sec, 400 read/sec). Use Base62 encoding on a distributed unique ID generator (Snowflake/DB auto-increment) to generate 7-character hashes. Cache frequent reads in Redis (LRU) and store mappings in a distributed key-value store (DynamoDB/Cassandra).",
     "Generate a random string in Python and save it in SQLite. If it crashes, get a bigger server.",
     "Use Base62 encoding to turn an ID into a short 7-character string. Store the mapping in a database like MongoDB or PostgreSQL, put Redis in front for caching popular URLs, and use a load balancer to scale web servers."),

    ("SWE_SECURITY_OWASP", "software_development", "Fullstack Developer", "Mid-Level",
     "What is Cross-Site Scripting (XSS) and SQL Injection (SQLi), and how do you protect web applications against them?",
     "XSS executes malicious client-side scripts in victim browsers; mitigated by context-aware output encoding, Content Security Policy (CSP), and HTTPOnly cookies. SQLi manipulates SQL queries via untrusted input; mitigated by parameterized queries/prepared statements and ORMs.",
     "XSS is when hackers steal your password. SQL injection is when they type admin into the login box. You fix it with an antivirus.",
     "SQL injection happens when user input is concatenated directly into SQL queries; prevent it with parameterized queries. XSS occurs when unescaped HTML/JavaScript runs in the user's browser; prevent it with HTML sanitization and CSP headers."),

    ("SWE_GIT_WORKFLOW", "software_development", "Software Engineer", "Junior",
     "What is the difference between git merge and git rebase, and when would you choose one over the other?",
     "Merge creates a non-destructive merge commit preserving exact historical commit timeline; rebase reapplies commits onto the tip of another branch creating a linear history. Use rebase for local feature branches before PRs; use merge for shared public branches to preserve historical integrity.",
     "Git merge and rebase are the exact same thing, rebase is just newer.",
     "Git merge keeps all history and adds a merge commit. Git rebase moves your branch commits on top of main for a clean linear history. Never rebase public shared branches."),

    ("SWE_MEMORY_LEAK", "software_development", "Backend Engineer", "Senior",
     "How do you profile, identify, and fix a memory leak in a production backend service?",
     "Monitor memory growth trends (heap allocation metrics). Capture heap snapshots/dumps at different intervals and compare object retention paths using profiling tools (pprof, Valgrind, Chrome DevTools). Common culprits: unclosed event listeners, unbounded global caches, and unreleased circular references.",
     "Restart the container whenever memory hits 90% using a cron job.",
     "Use profiling tools to take heap dumps over time and compare memory allocations. Look for objects that aren't being garbage collected, like growing global arrays or unremoved event listeners, and ensure connections are properly closed."),

    # AI & Data Science additions
    ("AI_BIAS_VARIANCE", "ai_data_science", "Data Scientist", "Junior",
     "Explain the bias-variance trade-off in machine learning and how it relates to underfitting and overfitting.",
     "Bias error arises from erroneous assumptions (underfitting, high bias). Variance error arises from sensitivity to small fluctuations in training data (overfitting, high variance). Total error = Bias^2 + Variance + Irreducible Error. Optimal models minimize total generalization error at the inflection point.",
     "Bias means the AI is racist and variance means the numbers vary a lot.",
     "High bias causes underfitting because the model is too simple to capture patterns. High variance causes overfitting because the model memorizes the training data. The goal is finding the balance between them."),

    ("AI_GRADIENT_DESCENT", "ai_data_science", "Machine Learning Engineer", "Junior",
     "Compare Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent. What role does Adam optimizer play?",
     "Batch GD calculates gradients across the entire dataset (stable but slow/memory-intensive). SGD computes gradient per single sample (fast, noisy, escapes local minima). Mini-batch GD balances both using mini-batches (32-512). Adam combines momentum (moving average of gradients) and RMSProp (scaling by squared gradients) for adaptive learning rates.",
     "Gradient descent goes downhill. Adam is a person who invented machine learning.",
     "Batch GD uses all data to update weights once. SGD updates on every single sample, which is noisy. Mini-batch GD updates on small batches like 32 or 64. Adam combines momentum with adaptive learning rates so you don't have to tune learning rate as much."),

    ("AI_DATA_LEAKAGE", "ai_data_science", "Data Scientist", "Mid-Level",
     "What is data leakage in machine learning, how does it happen, and how do you prevent it in your preprocessing pipeline?",
     "Data leakage occurs when information from outside the training dataset (target or future validation data) inadvertently influences model training. Common causes: fitting scalers/imputers before splitting, using future temporal data, or duplicate records. Prevent via strict Scikit-learn Pipelines, splitting before any preprocessing, and temporal cross-validation.",
     "Data leakage is when hackers steal your training CSV file. Prevent it with passwords.",
     "Data leakage happens when test data accidentally influences the training process, like fitting a StandardScaler on the whole dataset before train-test split. Prevent it by splitting first and using Pipeline objects to isolate transformations."),

    ("AI_PRECISION_RECALL_TRADE", "ai_data_science", "Data Scientist", "Junior",
     "Explain the mathematical relationship between precision and recall when tuning classification decision thresholds.",
     "Lowering the decision threshold classifies more instances as positive, increasing Recall (capturing more True Positives) but decreasing Precision (introducing more False Positives). Raising the threshold increases Precision at the expense of Recall. The F-measure or PR curve guides threshold selection.",
     "Precision and recall are the same thing just measured differently.",
     "Precision is TP/(TP+FP) and recall is TP/(TP+FN). If you lower the threshold, you catch more positives so recall goes up, but you get more false alarms so precision goes down. You balance them depending on the use case."),

    ("AI_RAG_VS_FINETUNE", "ai_data_science", "AI Engineer", "Senior",
     "When building an enterprise LLM solution, how do you decide between Retrieval-Augmented Generation (RAG) and Fine-Tuning?",
     "RAG is optimal for dynamic, rapidly updating factual knowledge, provenance/citation requirements, and private access control with low training compute. Fine-tuning is optimal for teaching specific output formats, stylistic tone, domain-specific vocabularies, or reducing prompt tokens. Enterprise architectures often combine both: fine-tuning for style/reasoning and RAG for fresh factual grounding.",
     "Fine-tuning is always better because you train the weights yourself. RAG is just prompt engineering.",
     "Use RAG when your data changes frequently or you need citations and access controls. Use fine-tuning when you want to change the model's behavior, tone, or make it follow a strict format. Often you can use both together."),

    ("AI_EMBEDDINGS_VECTOR_DB", "ai_data_science", "AI Engineer", "Mid-Level",
     "How do dense vector embeddings work, and how do approximate nearest neighbor (ANN) indexes like HNSW search them efficiently?",
     "Embeddings project semantic meaning into dense continuous vector spaces where cosine similarity reflects semantic closeness. Exact kNN requires O(N * D) brute force distance calculations. HNSW (Hierarchical Navigable Small World) constructs a multi-layer graph where greedy routing traverses upper sparse layers quickly, then refines locally in dense lower layers in O(log N) time.",
     "Vector databases store numbers in rows. They find similar words by matching letters.",
     "Embeddings convert text into lists of numbers representing meaning. HNSW builds a multi-layer graph of vectors so search doesn't have to check every vector, finding nearest neighbors in logarithmic time instead of scanning everything."),

    ("AI_HALLUCINATION_MITIGATION", "ai_data_science", "AI Engineer", "Mid-Level",
     "What causes hallucinations in Large Language Models, and what engineering methods reduce them in production?",
     "Hallucinations stem from LLMs being next-token probabilistic predictors optimizing surface plausibility rather than ground truth, exacerbated by training data noise and knowledge cutoffs. Mitigation: RAG with strict citation grounding, low temperature sampling, chain-of-thought verification, guardrail self-consistency checks, and structured output constraints.",
     "Hallucinations happen when the AI gets confused or tired. You fix it by telling the AI to never lie in the prompt.",
     "LLMs hallucinate because they predict the next likely token rather than checking facts. You can reduce hallucinations using RAG to ground responses in verified documents, setting temperature closer to 0, and using validation guardrails."),

    ("AI_DATA_DRIFT", "ai_data_science", "MLOps Engineer", "Mid-Level",
     "What is the difference between Data Drift and Concept Drift, and how do you monitor for them in production ML pipelines?",
     "Data Drift (Covariate Shift) is when input feature distribution P(X) changes while P(Y|X) remains constant. Concept Drift is when the relationship between features and target P(Y|X) changes (e.g., consumer behavior changes during inflation). Monitor via statistical tests (KS-test, PSI, Wasserstein distance) for features, and tracking performance metrics against delayed ground truth labels.",
     "Data drift is when the database moves to another cloud. Concept drift is when the CEO changes their mind.",
     "Data drift is when the input data distribution changes (P(X)). Concept drift is when the relationship between inputs and outputs changes (P(Y|X)). We monitor this using metrics like Population Stability Index (PSI) and tracking validation accuracy over time."),

    ("AI_CROSS_VALIDATION", "ai_data_science", "Data Scientist", "Junior",
     "Why is simple train/test split sometimes dangerous, and when should you use Stratified K-Fold or Time-Series Split?",
     "Simple split can introduce high variance or sampling bias on small datasets. Stratified K-Fold preserves class percentage ratios across all folds, essential for imbalanced classification. Time-Series Split (rolling window) prevents temporal data leakage by strictly training on past data and testing on future data.",
     "Train test split is dangerous because you might delete your test data. K-fold is just repeating the code 5 times.",
     "A simple split might accidentally put all rare class examples into the test set. Stratified K-Fold ensures each fold has the same class proportions. For time-series data, you must use Time-Series Split to avoid training on future data."),

    ("AI_TOKENIZATION_BPE", "ai_data_science", "NLP Engineer", "Mid-Level",
     "How does Byte-Pair Encoding (BPE) tokenization work in modern LLMs, and why is it preferred over character or word-level tokenization?",
     "BPE starts with a base vocabulary of individual characters/bytes and iteratively merges the most frequent adjacent symbol pairs into subword tokens until reaching vocabulary size V. Word-level tokenization suffers from massive vocabulary size and out-of-vocabulary (OOV) tokens; character-level results in excessive sequence lengths. BPE achieves optimal compression while handling rare/morphological words gracefully.",
     "Tokenization splits sentences on spaces. BPE is just a faster Python split function.",
     "BPE starts with characters and iteratively merges the most common pairs of tokens to build a subword vocabulary. It prevents out-of-vocabulary errors while keeping sequence lengths manageable compared to character-level tokenization."),

    # HR & Behavioural additions
    ("HR_LEADERSHIP_INFLUENCE", "hr_behavioural", "Engineering Lead", "Senior",
     "Tell me about a time you had to lead a project or influence team members without having formal managerial authority.",
     "STAR: Situation: Legacy monolith refactoring was deprioritized despite increasing developer friction. Task: Rally team to adopt modern modular patterns. Action: Built consensus through lunch-and-learn demos, created a clear migration blueprint, and partnered with tech lead. Result: Reduced build times by 35% with unanimous team buy-in.",
     "I told everyone what to do and complained to the director when they didn't listen.",
     "Our team had inconsistent code formatting. Since I wasn't the manager, I created a demo showing how Prettier and ESLint in pre-commit hooks saved code review time. The team liked it and we voted to adopt it across our repositories."),

    ("HR_ADAPTABILITY_PIVOT", "hr_behavioural", "All Roles", "Mid-Level",
     "Describe a situation where project requirements or business priorities changed drastically midway through your work. How did you adapt?",
     "STAR: Situation: Mid-way through building an in-house auth service, executive leadership mandated migrating to Okta. Task: Pivot smoothly without wasting sprint effort. Action: Identified salvageable components (RBAC schemas, audit logging), gracefully sunset in-progress modules, and reorganized sprint tickets. Result: Delivered Okta integration two weeks ahead of schedule.",
     "I was really angry that they wasted my work. I argued with the product manager for two days before finally doing what they asked.",
     "We were halfway through building a custom dashboard when client requirements changed to an integration with Salesforce. I paused development, met with the product manager to identify reusable logic, and adjusted the backlog. We delivered the new scope on time."),

    ("HR_CONSTRUCTIVE_FEEDBACK", "hr_behavioural", "All Roles", "Junior to Mid",
     "Tell me about a time you received critical or negative feedback from a manager or peer. How did you react and what actions did you take?",
     "STAR: Situation: During an annual review, my manager noted that while my code was high quality, my PR descriptions were terse, slowing code review velocity. Task: Improve engineering communication. Action: Created a comprehensive PR template with context, screenshots, and test plans, and proactively requested feedback. Result: Code review turnaround decreased by 40% and peer satisfaction scores rose.",
     "My manager gave me bad feedback because he didn't like me. I ignored it because my code was already working fine.",
     "In a review, my tech lead told me my pull requests were too large and difficult to review. I appreciated the feedback and started breaking features into smaller, atomic commits and smaller PRs. My team commented that reviewing my code became much easier."),

    ("HR_MENTORSHIP_GROWTH", "hr_behavioural", "Senior Engineer", "Senior",
     "Give an example of how you have mentored or helped develop a junior engineer or struggling teammate.",
     "STAR: Situation: An incoming junior engineer struggled with debugging complex asynchronous microservice failures. Task: Elevate their technical independence without micromanaging. Action: Established weekly pair programming sessions focusing on root-cause methodology (tracing, logs, breakpoints) rather than handing over answers; encouraged them to lead an internal tech talk. Result: The engineer became our top feature contributor within six months and was promoted to Mid-Level.",
     "I just wrote their code for them when they got stuck so the sprint wouldn't be delayed.",
     "A junior developer on my team was struggling with Git merge conflicts. I scheduled time to walk through how Git branches work under the hood and showed them visual diff tools. Within a month, they were resolving conflicts independently and helping new interns."),

    ("HR_ETHICAL_DILEMMA", "hr_behavioural", "All Roles", "Mid-Level",
     "Have you ever faced a situation where you were asked to cut corners or do something that compromised quality or integrity? How did you respond?",
     "STAR: Situation: Prior to a major product demo, a stakeholder asked to bypass data validation to quickly onboard a VIP client. Task: Protect user data integrity and compliance while supporting business goals. Action: Respectfully declined the bypass, clearly articulated the regulatory and security risks, and offered a fast-tracked verified onboarding script. Result: Protected customer security and retained the client without incidents.",
     "My boss told me to push code that had security bugs, so I just pushed it because he is the boss.",
     "Before a launch, we were pressured to skip integration testing. I explained that skipping tests on payment processing could cause transaction failures. I compromised by testing the critical payment path thoroughly while deferring nice-to-have cosmetic tests, ensuring we launched safely."),

    # Additional 15 Scenarios for rich coverage (~156 total records)
    ("SWE_REST_IDEMPOTENCY", "software_development", "Backend Engineer", "Mid-Level",
     "What does HTTP method idempotency mean, and why are POST, PUT, and DELETE treated differently in distributed architectures?",
     "An HTTP method is idempotent if making multiple identical requests has the same intended effect on the server as making a single request. GET, PUT, and DELETE are idempotent according to RFC 9110 (PUT replaces the exact resource state, DELETE ensures the resource is deleted). POST is non-idempotent because multiple calls create multiple distinct resources or trigger side-effects. In distributed networks with retries, non-idempotent POSTs require idempotency keys (e.g., UUID headers in Redis) to prevent duplicate transactions.",
     "Idempotency means a website is secure. GET and POST are the same except POST can send passwords.",
     "Idempotent means repeating the request produces the same result. PUT and DELETE are idempotent because updating or deleting multiple times leaves the system in the same state. POST is not idempotent because repeating it creates duplicates. To handle retries safely, we use idempotency keys."),

    ("SWE_WEB_PERFORMANCE", "software_development", "Frontend Developer", "Mid-Level",
     "What are Core Web Vitals (LCP, INP, CLS), and how do you optimize a modern single-page web application for them?",
     "Core Web Vitals are Google user experience metrics: Largest Contentful Paint (LCP < 2.5s) measures loading performance; Interaction to Next Paint (INP < 200ms) measures responsiveness to user actions; Cumulative Layout Shift (CLS < 0.1) measures visual stability. Optimization strategies include: route-level code splitting/lazy loading, preloading critical hero images, deferring non-critical scripts, serving modern WebP/AVIF images with explicit width/height dimensions, and breaking up long JavaScript main-thread tasks using web workers or requestIdleCallback.",
     "Web vitals just measure how fast your internet connection is. You fix it by telling users to use Chrome.",
     "Core Web Vitals include LCP for loading speed, INP for interaction responsiveness, and CLS for layout shifting. To optimize them, we compress images, use lazy loading for images and routes, specify image dimensions so the layout doesn't jump, and minimize heavy JavaScript bundles."),

    ("SWE_DATABASE_ACID", "software_development", "Backend Engineer", "Junior",
     "What are ACID properties in database management systems, and what happens when you reduce isolation levels?",
     "ACID guarantees database transaction reliability: Atomicity (all-or-nothing execution), Consistency (maintains defined schema invariants and constraints), Isolation (concurrent transactions execute independently without interference), Durability (committed changes persist across crashes). Reducing isolation levels from Serializable down to Repeatable Read, Read Committed, or Read Uncommitted trades strict correctness for concurrency throughput, exposing anomalies such as Dirty Reads, Non-Repeatable Reads, and Phantom Reads.",
     "ACID means the database runs fast and doesn't get corrupted when the power goes out.",
     "ACID stands for Atomicity, Consistency, Isolation, and Durability. Lowering the isolation level from Serializable allows higher throughput but can introduce anomalies like dirty reads or phantom reads."),

    ("SWE_LOAD_BALANCING", "software_development", "Cloud Engineer", "Mid-Level",
     "Compare Layer 4 vs Layer 7 load balancing and explain when you would use weighted round-robin versus least connections.",
     "Layer 4 (Transport layer, TCP/UDP) load balancers route packets based on IP and port without inspecting application payload, offering extreme throughput and lower CPU overhead. Layer 7 (Application layer, HTTP/HTTPS) load balancers inspect headers, cookies, and URIs, enabling path-based routing, SSL termination, and sticky sessions. Weighted round-robin distributes requests proportionally across heterogeneous hardware. Least connections is superior for long-lived, variable-duration transactions (e.g., WebSockets, file uploads) preventing worker starvation.",
     "Load balancers balance traffic. Layer 4 is 4 times faster than Layer 7. Round robin gives one request to each server.",
     "Layer 4 balances traffic at the TCP/IP level without reading HTTP content, making it very fast. Layer 7 inspects HTTP headers and URLs to do smart routing and SSL termination. Round robin is great for uniform requests, while least connections is better when requests take varying amounts of time to complete."),

    ("SWE_MESSAGE_QUEUES", "software_development", "Backend Engineer", "Mid-Level",
     "Why do distributed architectures decouple systems using message queues like RabbitMQ or Kafka, and how do you achieve at-least-once delivery?",
     "Message queues decouple microservices temporally and spatially, providing asynchronous processing, traffic spike smoothing (backpressure buffering), and fault isolation. At-least-once delivery is achieved through consumer acknowledgments (ACKs), persistent message disk logging, and retry mechanisms. Because messages may be redelivered upon consumer crash or timeout, consumers must be strictly idempotent (e.g., checking message IDs against a processed cache).",
     "Queues make code run in parallel. You don't need databases if you use a message queue.",
     "Message queues decouple services so the producer doesn't have to wait for the consumer, smoothing out traffic spikes. At-least-once delivery is achieved using message acknowledgments and retries, but consumers must be idempotent to handle duplicates safely."),

    ("AI_LOSS_FUNCTIONS", "ai_data_science", "Machine Learning Engineer", "Junior",
     "Compare Cross-Entropy Loss and Mean Squared Error (MSE). When is each appropriate and why?",
     "Cross-Entropy Loss (Log Loss) measures the divergence between predicted probability distributions and true categorical labels; it is appropriate for classification tasks because its gradient penalizes confident incorrect predictions exponentially without saturation. Mean Squared Error (MSE) measures the average squared difference between continuous predictions and targets; it is appropriate for regression problems under Gaussian noise assumptions. Using MSE for classification with sigmoid outputs causes vanishing gradients during backpropagation.",
     "MSE is for math and Cross-Entropy is for deep learning. You can use either one randomly.",
     "Cross-entropy loss is used for classification problems with softmax or sigmoid outputs because it works with probabilities. MSE is used for regression problems to predict continuous numbers. Using MSE for classification can lead to vanishing gradients."),

    ("AI_GRADIENT_VANISHING", "ai_data_science", "Deep Learning Engineer", "Mid-Level",
     "What causes the vanishing and exploding gradient problem in deep neural networks, and how do modern architectures solve it?",
     "Vanishing/exploding gradients occur during backpropagation when gradients are repeatedly multiplied across many layers using the chain rule. If weight matrices or activation derivatives have singular values < 1 (e.g., sigmoid/tanh derivatives <= 0.25), gradients decay exponentially toward zero; if > 1, they explode towards infinity. Solutions include: ReLU/GELU activations (constant derivative of 1 for positive values), residual skip connections (He et al., providing gradient highways d(x + f(x))/dx = 1 + f'(x)), Layer/Batch Normalization, proper weight initialization (He/Glorot), and gradient clipping.",
     "Vanishing gradient is when the computer runs out of memory. You fix it by decreasing batch size.",
     "Vanishing gradients happen when using activations like sigmoid whose derivatives are small, causing gradients to shrink to zero in early layers. Exploding gradients happen when weights are too large. We solve this using ReLU activations, residual connections (ResNets), Layer Normalization, and gradient clipping."),

    ("AI_PROMPT_ENGINEERING", "ai_data_science", "AI Engineer", "Junior",
     "What is Chain-of-Thought (CoT) prompting, and how does Few-Shot learning improve LLM reasoning accuracy?",
     "Chain-of-Thought (CoT) prompting encourages language models to decompose complex multi-step reasoning tasks into intermediate logical reasoning steps before outputting the final answer (e.g., 'Let's think step by step'). This allocates more computational compute tokens to reasoning, mitigating shortcut errors. Few-Shot prompting provides 2-5 structured input-output exemplars in the context window, establishing the required reasoning pattern, format schema, and stylistic tone via in-context learning without updating model weights.",
     "Chain of thought is when you chain multiple AI models together in a loop. Few shot means running the model three times.",
     "Chain-of-Thought prompting asks the model to explain its reasoning step by step before giving an answer, which improves accuracy on math and logic. Few-shot prompting provides a few examples of input and expected output in the prompt so the model learns the pattern in-context."),

    ("AI_FEATURE_ENGINEERING", "ai_data_science", "Data Scientist", "Mid-Level",
     "What is target encoding, what risk does it pose, and how do you prevent target leakage when using it?",
     "Target encoding replaces categorical levels with the expected value of the target variable for that category. While powerful for high-cardinality features compared to one-hot encoding, it poses a severe risk of overfitting and target leakage because target labels directly contaminate input features. To prevent leakage: apply target encoding inside cross-validation loops (calculating means strictly on training folds), add additive smoothing (m-estimate weighting category mean against global prior mean), or introduce out-of-fold Gaussian noise.",
     "Target encoding is when you train a model to predict the target. There is no risk.",
     "Target encoding replaces category values with the average target value for that category. The big risk is target leakage and overfitting. We prevent it by calculating encodings only within cross-validation folds and adding smoothing to blend category averages with the global mean."),

    ("AI_UNSUPERVISED_CLUSTERING", "ai_data_science", "Data Scientist", "Junior",
     "Compare K-Means clustering and DBSCAN. How do they differ in detecting non-spherical clusters and handling outliers?",
     "K-Means partitions data into K spherical Voronoi clusters by minimizing inertia (within-cluster sum of squares), requiring predefined K, sensitivity to scale, and extreme vulnerability to outliers. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points based on density reachability (eps distance and min_samples), discovering arbitrary-shaped non-linear clusters without specifying K, and explicitly designating low-density points as noise/outliers.",
     "K-means is supervised learning and DBSCAN is for images. Both find circles.",
     "K-Means requires you to specify the number of clusters (K) and assumes clusters are spherical, making it sensitive to outliers. DBSCAN is density-based, finds clusters of arbitrary shapes without pre-specifying K, and automatically labels sparse points as noise/outliers."),

    ("HR_WORKING_WITH_DIFFICULT", "hr_behavioural", "All Roles", "Mid-Level",
     "How do you handle working alongside a teammate who is consistently unresponsive or failing to deliver on dependencies?",
     "STAR: Situation: A critical microservice dependency owned by a remote colleague fell three weeks behind, blocking our frontend delivery. Task: Unblock the project without creating friction. Action: Scheduled an empathetic 1-on-1 to understand underlying blockers (discovered they were overwhelmed with unprioritized support tickets); helped them map out a minimal viable API contract and mocked responses so my team could proceed in parallel. Result: Unblocked frontend progress immediately, and colleague delivered the backend interface on schedule.",
     "I cc'ed their manager and VP on an angry email telling them they are ruining our sprint.",
     "When a teammate was slow to deliver an API my team needed, I reached out directly to check if they were blocked. I offered to help define a mock API schema so both teams could work independently. This removed the blocker and kept our relationship positive."),

    ("HR_AMBIGUITY_INITIATIVE", "hr_behavioural", "Software Engineer", "Mid-Level",
     "Tell me about a time you had to deliver a project where the requirements were vague or constantly changing.",
     "STAR: Situation: Tasked with building an 'automated onboarding intelligence dashboard' with only a one-sentence brief from leadership. Task: Clarify business value and convert ambiguity into actionable specifications. Action: Interviewed three department stakeholders to map pain points, created interactive Figma wireframes, established iterative two-week demo loops, and drafted an explicit product requirements document (PRD). Result: Delivered an MVP in four weeks adopted by 120 internal users with high NPS.",
     "I didn't do anything for two months because nobody gave me a detailed spec.",
     "When given a vague requirement to improve search results, I met with users to find their biggest pain points. I built a simple prototype with basic filtering, gathered feedback, and iterated. We improved search success rate by 25% without waiting for a perfect spec."),

    ("HR_WORK_LIFE_BURNOUT", "hr_behavioural", "All Roles", "Junior to Mid",
     "How do you maintain code quality and personal well-being when facing sustained periods of high workplace stress?",
     "I maintain sustainability through deliberate work boundaries, clear prioritization, and reliance on automated test suites. Rather than relying on willpower under fatigue, I rely on automated CI/CD guardrails, linters, and pair programming to catch errors early. I communicate capacity proactively with my manager, take restorative breaks, and practice time-blocking to protect deep work intervals.",
     "I don't get stressed. I just work all weekend and drink energy drinks.",
     "When work gets stressful, I protect quality by writing thorough unit tests and taking short mental breaks to prevent silly mistakes. I also communicate with my team lead early if sprint commitments are unrealistic to prevent burnout."),

    ("HR_DISAGREEMENT_MANAGER", "hr_behavioural", "All Roles", "Mid-Level",
     "Describe a time you disagreed with your manager's direction or decision. How did you communicate your perspective?",
     "STAR: Situation: My manager wanted to outsource a core search feature to a third-party SaaS with high recurring licensing fees. Task: Provide a thorough, respectful alternative analysis. Action: Formulated a comprehensive TCO (Total Cost of Ownership) comparison over 24 months, contrasting the SaaS cost against an in-house Elasticsearch implementation; presented the analysis objectively in a 1-on-1. Result: Manager appreciated the data-backed view; we chose in-house and saved $80,000 annually.",
     "I told my manager his decision was stupid in front of the whole department meeting.",
     "My manager suggested delaying writing automated tests until after launch. In our 1-on-1, I explained that skipping tests on the billing module would create high financial risk. I proposed writing tests only for the core payment flow first. He agreed with this balanced approach."),

    ("HR_CONTINUOUS_LEARNING", "hr_behavioural", "All Roles", "Junior",
     "How do you stay updated with rapidly evolving technology, and can you share something new you recently learned?",
     "I stay current through a combination of reading engineering blogs (e.g., Uber, Netflix tech blogs), studying official documentation and research papers (arXiv), building hands-on open-source side projects, and participating in tech communities. Recently, I explored Parameter-Efficient Fine-Tuning (LoRA/QLoRA), learning how low-rank matrix decomposition enables fine-tuning multi-billion parameter models on consumer GPUs.",
     "I don't study outside of work hours because college taught me everything I need.",
     "I stay updated by reading tech blogs, Hacker News, and following open-source projects on GitHub. Recently I learned about QLoRA and how 4-bit quantization allows fine-tuning language models on a single GPU.")
]


def expand_additional_scenarios() -> List[Dict[str, Any]]:
    """Converts the shorthand questions into full 3-tier scenarios."""
    expanded = []
    for item in ADDITIONAL_QUESTIONS:
        prefix, cat, role, level, question, ideal_strong, weak_ans, avg_ans = item
        scenario = {
            "id_prefix": prefix,
            "category": cat,
            "role": role,
            "experience_level": level,
            "question": question,
            "weak": {
                "answer": weak_ans,
                "rubric": {"relevance": 5, "accuracy_depth": 3, "clarity_structure": 4, "completeness": 3},
                "score": 3 + (hash(prefix) % 2),
                "strengths": ["Attempts to answer the question directly."],
                "weaknesses": [
                    "Lacks technical depth or structured methodology.",
                    "Fails to address trade-offs, edge cases, or concrete execution."
                ],
                "advice": [
                    "Provide concrete technical definitions and structured mechanisms.",
                    "Explain the 'why' behind architectural choices rather than just stating opinions."
                ],
                "improved": ideal_strong,
                "follow_up": f"Could you provide a concrete production example where you applied this principle?"
            },
            "average": {
                "answer": avg_ans,
                "rubric": {"relevance": 8, "accuracy_depth": 7, "clarity_structure": 7, "completeness": 7},
                "score": 7,
                "strengths": [
                    "Understands the fundamental concepts and communicates clearly.",
                    "Identifies practical advantages and common tooling."
                ],
                "weaknesses": [
                    "Could elaborate further on deep system failure modes or edge-case handling.",
                    "Misses quantitative benchmarks or specific architectural patterns."
                ],
                "advice": [
                    "Quantify system metrics (throughput, latency, error budgets).",
                    "Discuss failure recovery or operational trade-offs explicitly."
                ],
                "improved": ideal_strong,
                "follow_up": f"What are the operational challenges or failure modes associated with this approach at scale?"
            },
            "strong": {
                "answer": ideal_strong,
                "rubric": {"relevance": 10, "accuracy_depth": 10, "clarity_structure": 10, "completeness": 10},
                "score": 10,
                "strengths": [
                    "Comprehensive, technically precise, and articulates clear trade-offs.",
                    "Structured logically and highlights production best practices."
                ],
                "weaknesses": ["None. The response is authoritative and well-rounded."],
                "advice": [
                    "Continue pairing high-level architecture with concrete implementation trade-offs."
                ],
                "improved": ideal_strong,
                "follow_up": f"How would you instrument metrics and observability to monitor this in a production cluster?"
            }
        }
        expanded.append(scenario)
    return expanded


def build_raw_dataset() -> List[Dict[str, Any]]:
    """Compiles all scenarios and their weak/average/strong variants into records."""
    all_scenarios = SCENARIOS + expand_additional_scenarios()
    dataset: List[Dict[str, Any]] = []

    for sc in all_scenarios:
        for quality in ["weak", "average", "strong"]:
            data_variant = sc[quality]
            record_id = f"{sc['id_prefix']}_{quality.upper()}"

            record = {
                "id": record_id,
                "category": sc["category"],
                "role": sc["role"],
                "experience_level": sc["experience_level"],
                "instruction": (
                    "You are InterviewMate, an expert AI mock interviewer and technical evaluator. "
                    "Analyze the candidate's interview answer below according to the scoring rubric. "
                    "Evaluate relevance, technical accuracy/depth, clarity, and completeness. "
                    "Provide an overall score out of 10, detailed strengths, weaknesses, actionable advice, "
                    "an improved sample answer, and a contextual follow-up question in strict JSON format."
                ),
                "interview_question": sc["question"],
                "candidate_answer": data_variant["answer"],
                "answer_quality": quality,
                "scoring_rubric": data_variant["rubric"],
                "overall_score": data_variant["score"],
                "strengths": data_variant["strengths"],
                "weaknesses": data_variant["weaknesses"],
                "actionable_feedback": data_variant["advice"],
                "improved_sample_answer": data_variant["improved"],
                "follow_up_question": data_variant["follow_up"]
            }
            dataset.append(record)

    return dataset


def validate_record(record: Dict[str, Any], idx: int) -> None:
    """Strictly validates the schema, field types, and constraints of a record."""
    required_keys = [
        "id", "category", "role", "experience_level", "instruction",
        "interview_question", "candidate_answer", "answer_quality",
        "scoring_rubric", "overall_score", "strengths", "weaknesses",
        "actionable_feedback", "improved_sample_answer", "follow_up_question"
    ]
    for key in required_keys:
        if key not in record:
            raise ValueError(f"Record {idx} ({record.get('id', 'unknown')}) missing required key: {key}")

    # Check non-empty strings
    str_keys = ["id", "category", "role", "instruction", "interview_question",
                "candidate_answer", "improved_sample_answer", "follow_up_question"]
    for key in str_keys:
        val = record[key]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Record {record['id']}: field '{key}' must be a non-empty string.")

    # Check score ranges
    score = record["overall_score"]
    if not isinstance(score, (int, float)) or not (1 <= score <= 10):
        raise ValueError(f"Record {record['id']}: overall_score must be between 1 and 10, got {score}")

    rubric = record["scoring_rubric"]
    rubric_keys = ["relevance", "accuracy_depth", "clarity_structure", "completeness"]
    for rk in rubric_keys:
        if rk not in rubric or not (1 <= rubric[rk] <= 10):
            raise ValueError(f"Record {record['id']}: rubric '{rk}' must be between 1 and 10, got {rubric.get(rk)}")

    # Check lists of strings
    list_keys = ["strengths", "weaknesses", "actionable_feedback"]
    for lk in list_keys:
        val = record[lk]
        if not isinstance(val, list) or len(val) == 0:
            raise ValueError(f"Record {record['id']}: '{lk}' must be a non-empty list.")
        for item in val:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Record {record['id']}: '{lk}' contains empty item: {item}")


def generate_and_save_dataset():
    """Generates, validates, and partitions the dataset into train and eval splits."""
    os.makedirs(DATA_DIR, exist_ok=True)
    records = build_raw_dataset()

    # Validate all records
    for i, r in enumerate(records):
        validate_record(r, i)

    # Group by question scenario to guarantee NO data leakage between train and eval!
    # All 3 variants (weak, average, strong) of a question must stay together in either train or eval.
    scenarios_dict: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        prefix = r["id"].rsplit("_", 1)[0]
        if prefix not in scenarios_dict:
            scenarios_dict[prefix] = []
        scenarios_dict[prefix].append(r)

    scenario_keys = list(scenarios_dict.keys())
    # Deterministic shuffle
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(scenario_keys)

    # 80% train, 20% eval split based on whole question scenarios
    split_idx = int(len(scenario_keys) * 0.8)
    train_prefixes = set(scenario_keys[:split_idx])
    eval_prefixes = set(scenario_keys[split_idx:])

    train_records = [r for r in records if r["id"].rsplit("_", 1)[0] in train_prefixes]
    eval_records = [r for r in records if r["id"].rsplit("_", 1)[0] in eval_prefixes]

    # Write training jsonl
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write evaluation jsonl
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("InterviewMate Dataset Generation Completed Successfully!")
    print("=" * 60)
    print(f"Total Unique Scenarios: {len(scenario_keys)}")
    print(f"Total Records Generated: {len(records)}")
    print(f"Training Records Saved:   {len(train_records)} -> {TRAIN_FILE}")
    print(f"Evaluation Records Saved: {len(eval_records)} -> {EVAL_FILE}")
    print("-" * 60)

    # Breakdown by category
    cats: Dict[str, int] = {}
    qualities: Dict[str, int] = {}
    for r in records:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
        qualities[r["answer_quality"]] = qualities.get(r["answer_quality"], 0) + 1

    print("Category Breakdown:")
    for c, count in cats.items():
        print(f"  - {c}: {count} records")

    print("\nQuality Breakdown:")
    for q, count in qualities.items():
        print(f"  - {q}: {count} records")

    print("-" * 60)
    print("Notice: Synthetic data generated using expert rubrics.")
    print("Zero data leakage: All evaluation scenarios are strictly held out from training.")
    print("=" * 60)


if __name__ == "__main__":
    generate_and_save_dataset()
