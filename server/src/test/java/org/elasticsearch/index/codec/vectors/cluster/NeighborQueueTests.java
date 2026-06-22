/*
 * @notice
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modifications copyright (C) 2024 Elasticsearch B.V.
 */
package org.elasticsearch.index.codec.vectors.cluster;

import org.elasticsearch.test.ESTestCase;

/**
 * copied and modified from Lucene
 */
public class NeighborQueueTests extends ESTestCase {
    public void testNeighborsProduct() {
        // make sure we have the sign correct
        NeighborQueue nn = new NeighborQueue(2, false);
        assertTrue(nn.insertWithOverflow(2, 0.5f));
        assertTrue(nn.insertWithOverflow(1, 0.2f));
        assertTrue(nn.insertWithOverflow(3, 1f));
        assertEquals(0.5f, nn.topScore(), 0);
        nn.pop();
        assertEquals(1f, nn.topScore(), 0);
        nn.pop();
    }

    public void testNeighborsMaxHeap() {
        NeighborQueue nn = new NeighborQueue(2, true);
        assertTrue(nn.insertWithOverflow(2, 2));
        assertTrue(nn.insertWithOverflow(1, 1));
        assertFalse(nn.insertWithOverflow(3, 3));
        assertEquals(2f, nn.topScore(), 0);
        nn.pop();
        assertEquals(1f, nn.topScore(), 0);
    }

    public void testTopMaxHeap() {
        NeighborQueue nn = new NeighborQueue(2, true);
        nn.add(1, 2);
        nn.add(2, 1);
        // lower scores are better; highest score on top
        assertEquals(2, nn.topScore(), 0);
        assertEquals(1, nn.topNode());
    }

    public void testTopMinHeap() {
        NeighborQueue nn = new NeighborQueue(2, false);
        nn.add(1, 0.5f);
        nn.add(2, -0.5f);
        // higher scores are better; lowest score on top
        assertEquals(-0.5f, nn.topScore(), 0);
        assertEquals(2, nn.topNode());
    }

    public void testClear() {
        NeighborQueue nn = new NeighborQueue(2, false);
        nn.add(1, 1.1f);
        nn.add(2, -2.2f);
        nn.clear();

        assertEquals(0, nn.size());
    }

    public void testMaxSizeQueue() {
        NeighborQueue nn = new NeighborQueue(2, false);
        nn.add(1, 1);
        nn.add(2, 2);
        assertEquals(2, nn.size());
        assertEquals(1, nn.topNode());

        // insertWithOverflow does not extend the queue
        nn.insertWithOverflow(3, 3);
        assertEquals(2, nn.size());
        assertEquals(2, nn.topNode());

        // add does extend the queue beyond maxSize
        nn.add(4, 1);
        assertEquals(3, nn.size());
    }

    public void testUnboundedQueue() {
        NeighborQueue nn = new NeighborQueue(1, true);
        float maxScore = -2;
        int maxNode = -1;
        for (int i = 0; i < 256; i++) {
            // initial size is 32
            float score = random().nextFloat();
            if (score > maxScore) {
                maxScore = score;
                maxNode = i;
            }
            nn.add(i, score);
        }
        assertEquals(maxScore, nn.topScore(), 0);
        assertEquals(maxNode, nn.topNode());
    }

    public void testInvalidArguments() {
        expectThrows(IllegalArgumentException.class, () -> new NeighborQueue(0, false));
    }

    public void testToString() {
        assertEquals("Neighbors[0]", new NeighborQueue(2, false).toString());
    }

    public void testPopRawAndAddRawReturnsRawWhenNewTop() {
        NeighborQueue nn = new NeighborQueue(2, false);
        long first = nn.encode(1, 1.0f);
        long second = nn.encode(2, 2.0f);
        nn.insertWithOverflow(first);
        nn.insertWithOverflow(second);

        long newTop = nn.encode(3, 0.5f);
        assertTrue(newTop < nn.peek());

        long result = nn.popRawAndAddRaw(newTop);
        assertEquals(newTop, result);
        assertEquals(2, nn.size());
        assertEquals(first, nn.peek());
    }

    public void testUpdateTopDemotesWhenScoreLowered() {
        // max-heap: highest score on top. Lowering the top's score below a sibling must demote it,
        // exactly as a pop()+add() of the new score would, but in a single sift-down.
        NeighborQueue nn = new NeighborQueue(3, true);
        nn.add(1, 0.9f);
        nn.add(2, 0.8f);
        nn.add(3, 0.7f);
        assertEquals(1, nn.topNode());
        assertEquals(0.9f, nn.topScore(), 0);

        // Refine node 1: its combined score (0.65) drops below nodes 2 and 3.
        nn.updateTop(1, 0.65f);
        assertEquals(3, nn.size());
        assertEquals(2, nn.topNode());
        assertEquals(0.8f, nn.topScore(), 0);

        nn.pop();
        assertEquals(3, nn.topNode());
        assertEquals(0.7f, nn.topScore(), 0);
        nn.pop();
        assertEquals(1, nn.topNode());
        assertEquals(0.65f, nn.topScore(), 0);
    }

    public void testUpdateTopKeepsTopWhenStillBest() {
        // When the replacement score keeps the element as the maximum, the single sift-down must
        // leave it on top (a refined score that is still the best need not move).
        NeighborQueue nn = new NeighborQueue(3, true);
        nn.add(1, 0.9f);
        nn.add(2, 0.8f);
        nn.add(3, 0.7f);
        assertEquals(1, nn.topNode());

        nn.updateTop(1, 0.85f);
        assertEquals(3, nn.size());
        assertEquals(1, nn.topNode());
        assertEquals(0.85f, nn.topScore(), 0);
    }

    public void testUpdateTopMatchesPopThenAdd() {
        // Property: re-scoring the top via updateTop yields the same heap contents (same drained
        // order) as an explicit popRaw()+add() of the new score. Randomized to guard the invariant.
        final int n = randomIntBetween(2, 64);
        NeighborQueue viaUpdate = new NeighborQueue(n, true);
        NeighborQueue viaPopAdd = new NeighborQueue(n, true);
        for (int i = 0; i < n; i++) {
            final float score = random().nextFloat();
            viaUpdate.add(i, score);
            viaPopAdd.add(i, score);
        }
        final int topNode = viaUpdate.topNode();
        // A refined score never exceeds the prefix score it replaces, so lower the top's score.
        final float lowered = viaUpdate.topScore() - random().nextFloat();

        viaUpdate.updateTop(topNode, lowered);

        final long raw = viaPopAdd.popRaw();
        assertEquals(topNode, viaPopAdd.decodeNodeId(raw));
        viaPopAdd.add(topNode, lowered);

        assertEquals(viaPopAdd.size(), viaUpdate.size());
        while (viaUpdate.size() > 0) {
            assertEquals(viaPopAdd.topNode(), viaUpdate.topNode());
            assertEquals(viaPopAdd.topScore(), viaUpdate.topScore(), 0);
            viaPopAdd.pop();
            viaUpdate.pop();
        }
    }

}
