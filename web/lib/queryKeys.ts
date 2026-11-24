// Query keys that more than one file has to agree on.
//
// Most keys are the api method's own name (["datasets"], ["dataset", id]) and need no entry here: the
// convention is the registry. This file is for the exceptions, where a key is not derivable from a method
// name and a page that invalidates it would otherwise have to copy a long string and hope.

/**
 * The review queue's six reads, fetched and cached as one because its tabs are read against each other
 * (app/review/queue/page.tsx). Anything that confirms or dismisses an error candidate, rules on a recall
 * candidate or changes what is pending invalidates this.
 */
export const REVIEW_QUEUE_KEY =
  "alScore+errorCandidates+recallCandidates+recallProgress+controlPending+governPrecision";
