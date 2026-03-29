// ---- Catalog types (from /api/catalog) ----
export interface Scenario {
  id: string;
  title: string;
  subtitle: string;
  business_title?: string;
  industry_problem?: string;
  decision_focus?: string;
  description: string;
  prompt: string;
  icon: string;
  category?: string;
}

export interface ModuleInfo {
  description: string;
  domain: string;
  config_keys: Record<string, string[]>;
  defaults: Record<string, unknown>;
}

export interface Capability {
  name: string;
  icon: string;
  status: 'Full' | 'Partial' | 'None';
}

export interface Catalog {
  scenarios: Scenario[];
  modules: Record<string, ModuleInfo>;
  tools: string[];
  capabilities: Capability[];
}

// ---- SSE event types ----
export interface TurnStartEvent {
  turn_id: string;
  prompt: string;
}

export interface PhaseEvent {
  phase: string;
  message: string;
}

export interface ThinkingEvent {
  thinking: string;
  plan: string[];
  source: 'llm' | 'heuristic';
}

export interface ToolStartEvent {
  index: number;
  tool: string;
  args: Record<string, unknown>;
}

export interface FieldStats {
  shape?: number[];
  size?: number;
  min: number;
  max: number;
  mean: number;
}

export interface ToolResultPayload {
  module_name?: string;
  success?: boolean;
  error?: string;
  elapsed_time_s?: number;
  _elapsed_s?: number;
  fields_stats?: Record<string, FieldStats>;
  state?: {
    fields?: Record<string, { data: number[] | number[][]; unit?: string }>;
  };
  // sweep
  param_path?: string;
  results?: Array<{
    param_value: number;
    success: boolean;
    fields: Record<string, FieldStats>;
  }>;
  values?: number[];
  // optimize
  optimal_param_value?: number;
  optimal_metric?: number;
  objective_field?: string;
  objective?: string;
  bounds?: number[];
  evaluations?: Array<{ param_value: number; metric: number }>;
  // compare
  comparison?: Array<{
    label: string;
    success: boolean;
    fields: Record<string, FieldStats>;
  }>;
  // UQ
  n_samples?: number;
  statistics?: Record<string, {
    mean: number;
    std: number;
    cv_percent: number;
    p5: number;
    p95: number;
  }>;
  // chain
  chain_results?: Array<{
    step: number;
    step_label?: string;
    module_name?: string;
    success?: boolean;
    error?: string;
  }>;
  // list_modules / describe_module
  modules?: Record<string, { description: string }>;
  domain?: string;
  config_keys?: Record<string, string[]>;
  [key: string]: unknown;
}

export interface ToolProgressEvent {
  index: number;
  tool: string;
  step: number;
  total: number;
  detail: Record<string, unknown>;
}

export interface ToolResultEvent {
  index: number;
  tool: string;
  elapsed_s: number;
  success: boolean;
  result: ToolResultPayload;
}

export interface SummaryEvent {
  summary: string;
  insights: string;
}

export interface ReflectionEvent {
  step: number;
  reason: string;
}

export interface ReflectionAddendumEvent {
  addendum: string;
}

export interface ChitchatEvent {
  response: string;
}

export interface TurnCompleteEvent {
  turn_id: string;
  total_tools: number;
  all_succeeded: boolean;
  goal_driven?: boolean;
  goal_satisfied?: boolean;
  answer?: number;
}

export interface ErrorEvent {
  error: string;
}

// ---- Goal-driven convergence events ----
export interface GoalExtractedEvent {
  goal: {
    goal_type: string;
    metric: string;
    operator?: string;
    threshold?: number;
    unit?: string;
    search_variable?: string;
    search_bounds?: number[];
    module_name?: string;
  };
  message: string;
}

export interface AnalyticalEstimateEvent {
  estimate: {
    estimate?: number;
    unit?: string;
    governing_equation?: string;
    suggested_bounds?: number[];
    confidence: string;
    reasoning: string;
  };
  message: string;
}

export interface ConvergenceStepEvent {
  iteration: number;
  max_iterations: number;
  evaluation: {
    satisfied: boolean;
    answer?: number;
    answer_unit?: string;
    accuracy?: string;
    accuracy_pct?: number;
    closest_value?: number;
    gap_ratio?: number;
    action: string;
    details: string;
  };
  total_solver_runs: number;
}

export interface GoalSatisfiedEvent {
  answer: number;
  answer_unit?: string;
  accuracy?: string;
  accuracy_pct?: number;
  iterations_used: number;
  total_solver_runs: number;
  details: string;
}

export interface ConvergenceAdaptingEvent {
  iteration: number;
  action: string;
  details: string;
  next_bounds?: number[];
}

export interface ConvergenceStalledEvent {
  iteration: number;
  reason: string;
  best_so_far: Record<string, unknown>;
}

export interface ConvergenceExhaustedEvent {
  iterations_used: number;
  total_solver_runs: number;
  best_so_far: Record<string, unknown>;
}

// ---- Chat message model ----
export type SSEBlock =
  | { type: 'phase'; data: PhaseEvent }
  | { type: 'thinking'; data: ThinkingEvent }
  | { type: 'tool_start'; data: ToolStartEvent }
  | { type: 'tool_progress'; data: ToolProgressEvent }
  | { type: 'tool_result'; data: ToolResultEvent }
  | { type: 'summary'; data: SummaryEvent }
  | { type: 'reflection'; data: ReflectionEvent }
  | { type: 'reflection_addendum'; data: ReflectionAddendumEvent }
  | { type: 'chitchat'; data: ChitchatEvent }
  | { type: 'turn_complete'; data: TurnCompleteEvent }
  | { type: 'error'; data: ErrorEvent }
  | { type: 'goal_extracted'; data: GoalExtractedEvent }
  | { type: 'analytical_estimate'; data: AnalyticalEstimateEvent }
  | { type: 'convergence_step'; data: ConvergenceStepEvent }
  | { type: 'goal_satisfied'; data: GoalSatisfiedEvent }
  | { type: 'convergence_adapting'; data: ConvergenceAdaptingEvent }
  | { type: 'convergence_stalled'; data: ConvergenceStalledEvent }
  | { type: 'convergence_exhausted'; data: ConvergenceExhaustedEvent };

export interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  text?: string;           // user messages
  blocks: SSEBlock[];      // agent messages built from SSE events
}
