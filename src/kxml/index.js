export { parseKXML }                                        from './kxml-parser.js';
export { PhaseGatedDispatcher, PHASE_ORDER }                from './kxml-dispatcher.js';
export { KXMLGraph }                                        from './kxml-graph.js';
export {
  OPS, dispatchOp,
  geodesicDist, parallelTransport, ricciFlowStep,
  geometricAttention, foldCompress, crossEntropy,
} from './kxml-ops.js';
export {
  ShardRegistry,
  PHASE_RESIDENCY,
  KXML_FOLD_TO_SCXQ2,
  SCXQ2_OPCODES,
  SCXQ2_DOMAIN,
  SCXQ2_FOLD_ID,
} from './kxml-shard-registry.js';
