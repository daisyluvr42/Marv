import { configureStore } from "@reduxjs/toolkit";
import coworkReducer from "./slices/coworkSlice";
import imReducer from "./slices/imSlice";
import modelReducer from "./slices/modelSlice";
import quickActionReducer from "./slices/quickActionSlice";
import scheduledTaskReducer from "./slices/scheduledTaskSlice";
import skillReducer from "./slices/skillSlice";

export const store = configureStore({
  reducer: {
    model: modelReducer,
    cowork: coworkReducer,
    skill: skillReducer,
    im: imReducer,
    quickAction: quickActionReducer,
    scheduledTask: scheduledTaskReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
